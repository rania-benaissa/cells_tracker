import numpy as np
from cells_detection.cellConfig import *
from cells_detection.train_model import selectWeights
from cells_detection.cell import *
import mrcnn.model as modellib
from setup import *
from cells_tracking.frame import *
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mrcnn.visualize import draw_frame_boxes, random_colors
import matplotlib.animation as animation
from mrcnn.utils import compute_overlaps
from cells_tracking.object import *
from IPython.display import HTML
from skimage.measure import regionprops

from skimage.color import gray2rgb


def rescale_bounding_boxes(bounding_boxes, original_shape, new_shape):
    original_width, original_height = original_shape[1], original_shape[0]
    new_width, new_height = new_shape[1], new_shape[0]
    width_scale = new_width / original_width
    height_scale = new_height / original_height
    rescaled_bounding_boxes = []
    for y1, x1, y2, x2 in bounding_boxes:
        x1 *= width_scale
        y1 *= height_scale
        x2 *= width_scale
        y2 *= height_scale
        rescaled_bounding_boxes.append([y1, x1, y2, x2])
    return np.array(rescaled_bounding_boxes)


def get_ax(rows=1, cols=1, size=12):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(
        size * cols, (size - 5) * rows))
    plt.tight_layout()
    return fig, ax


def prepareDataset(dataset_dir, subdir="val"):

    # Validation dataset
    dataset_test = CellDataset()
    dataset_test.load_cell(dataset_dir, subdir)
    dataset_test.prepare()

    return dataset_test


def extract_image_features(image_id, model, config, dataset, feature_size=64):

    image = dataset.load_image(image_id)

    mrcnn = model.run_graph([image], [
        # backbone output
        ("res4f_out", model.keras_model.get_layer("res4f_out").output),  #
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])

    # (1024, 88, 140)
    activations = np.transpose(mrcnn["res4f_out"][0, :, :, :], [2, 0, 1])

    # resize activations

    activations = np.array([np.squeeze(utils.resize_image(
        np.expand_dims(activ_map, -1),
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)[0]) for activ_map in activations])

    nb_detections = mrcnn['detections'][0, :, 4].astype(np.int32).sum()

    detected_boxes = utils.denorm_boxes(
        mrcnn["detections"][0, :nb_detections, :4], activations[0].shape)

    # (nb_objs, nb_features,feature_sizexfeature_size)
    # objects_features = np.zeros(
    #     (nb_detections, len(activations), feature_size**2))

    objects_features = np.zeros(
        (nb_detections, feature_size**2))

    # for each (88, 140)
    for j in range(len(detected_boxes)):

        y1, x1, y2, x2 = detected_boxes[j]
        x = (feature_size - (x2 - x1)) // 2
        y = (feature_size - (y2 - y1)) // 2

        object_feature = np.zeros((feature_size, feature_size))

        for i in range(len(activations)):
            # it s one of the activation maps of the image

            # object_feature = np.zeros((feature_size, feature_size))

            # Paste the foreground image onto the background image
            # object_feature[y:y + (y2 - y1), x:x + (x2 - x1)
            #                ] = activations[i][y1:y2, x1:x2]

            object_feature[y:y + (y2 - y1), x:x + (x2 - x1)
                           ] += activations[i][y1:y2, x1:x2]

            # objects_features[j, i] = object_feature.ravel()
        objects_features[j] = object_feature.ravel()

    # HERE I RETURN THE BOXES FOR THE IMAGE
    image, _, _, _, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    detected_boxes = utils.denorm_boxes(
        mrcnn["detections"][0, :nb_detections, :4], (np.squeeze(image).shape[:2]))

    return image, objects_features, detected_boxes


def extract_image_boxes(image_id, model, config, dataset):

    image = dataset.load_image(image_id)

    mrcnn = model.run_graph([image], [
        # backbone output
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])

    nb_detections = mrcnn['detections'][0, :, 4].astype(np.int32).sum()

    # HERE I RETURN THE BOXES FOR THE IMAGE
    image, _, _, _, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    detected_boxes = utils.denorm_boxes(
        mrcnn["detections"][0, :nb_detections, :4], (np.squeeze(image).shape[:2]))

    return image, detected_boxes


def normalize(matrix):
    # Normalize the matrix so that the minimum value is 0 and the maximum value is 1
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    return matrix


def remove_disappearing_objects(cost_matrix):

    removed_rows = np.where((cost_matrix == 0).all(axis=1))[0]

    # remove those rows from cost matrix
    cost_matrix = np.delete(cost_matrix, removed_rows, axis=0)

    return 1 - cost_matrix, removed_rows


class Tracker():

    def __init__(self, DATASET_DIR2="", weights="", subdir="val", nb_images="all"):

        self.gt_frames = None

        self.SEQUENCE_DIR = "Images" if DATASET_DIR2 == "" else DATASET_DIR2

        self.detection_weights = "train_results3/cell20230123T1638/mask_rcnn_cell_0080.h5" if weights == "" else weights

        print("Weights: ", self.detection_weights)
        print("Dataset: ", self.SEQUENCE_DIR)

        # # Configurations

        self.detection_config = CellInferenceConfig()

        self.dataset = prepareDataset(self.SEQUENCE_DIR, subdir)

        # # Create model
        self.detection_model = modellib.MaskRCNN(mode="inference", config=self.detection_config,
                                                 model_dir=TEST_SAVE_DIR)

        # select weights
        self.detection_model = selectWeights(
            self.detection_model, self.detection_weights)

        # this sets the self.frames
        self.fill_sequence(self.dataset, nb_images)

        self.colors = random_colors(1000)

        self.max_id = 0

    def fill_sequence(self, dataset, nb_images):

        self.frames = np.empty((len(dataset.image_ids)), dtype=object) if(
            nb_images == "all") else np.empty((nb_images), dtype=object)

        images_ids = dataset.image_ids if(
            nb_images == "all")else dataset.image_ids[:nb_images]

        for i in range(len(images_ids)):

            identifier = dataset.image_info[images_ids[i]]["id"]

            image, image_boxes = extract_image_boxes(
                images_ids[i], self.detection_model, self.detection_config, dataset)

            self.frames[i] = Frame(
                identifier, image, image_boxes=image_boxes, idx=i)

    def adjust_matchs(self, frame1, frame2, threshold=50):

        frame1_removed_centroids, frame1_removed_ids = frame1.get_known_objects(
            self.removed_rows)

        frame2_unknown_centroids, positions = frame2.get_unknown_objects()

        if(len(frame1_removed_centroids) != 0 and len(frame2_unknown_centroids) != 0):

            distance = cdist(
                frame1_removed_centroids, frame2_unknown_centroids)
            # print(frame1_removed_ids)
            # print(distance)

            for i in range(len(distance)):

                min_val = distance[i].min()

                if(min_val < threshold):

                    idx = positions[distance[i].argmin()]

                    frame2.objects[idx].id = frame1_removed_ids[i]
                    frame2.objects[idx].isNew = False

        return frame2

    def update_ids(self, frame1, frame2, row_ind, col_ind):

        frame1_ids = np.delete(frame1.get_all_ids(), self.removed_rows, axis=0)

        # it will have min match anyways
        for i in range(len(col_ind)):

            frame2.objects[col_ind[i]].id = frame1_ids[row_ind[i]]
            frame2.objects[col_ind[i]].isNew = False

        self.max_id = self.max_id if self.max_id >= frame2.get_max_id() else frame2.get_max_id()

        # here i'm supposing that
        #  the removed ids of th first frame are moving very fast
        # so i check its distance with the non-identified

        frame2 = self.adjust_matchs(frame1, frame2)

        for obj in frame2.objects:

            if obj.id == -1:

                obj.id = self.max_id + 1
                self.max_id += 1

    def track_objects(self):

        # for second frame
        cost_matrix, self.removed_rows = self.cost_matrix(
            self.frames[0], self.frames[1])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # get the max id already given
        self.max_id = len(self.frames[0].objects)

        self.update_ids(self.frames[0], self.frames[1], row_ind, col_ind)

        for i in range(2, len(self.frames)):

            cost_matrix, self.removed_rows = self.cost_matrix(
                self.frames[i - 1], self.frames[i])

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            self.update_ids(self.frames[i - 1],
                            self.frames[i], row_ind, col_ind)

    def cost_matrix(self, frame1, frame2):

        # add positional information (positional features)

        iou_cost_matrix = compute_overlaps(
            frame1.get_all_boxes(), frame2.get_all_boxes())

        return remove_disappearing_objects(iou_cost_matrix)

    def show_tracking(self):

        for i in range(len(self.frames)):

            fig, ax = get_ax(cols=2)

            objects_ids = self.frames[i].get_all_ids()

            objects_boxes = self.frames[i].get_all_boxes()

            captions = list(map(str, objects_ids))

            colors = [self.colors[i] for i in objects_ids]

            title = "Nb. objects = " + \
                str(len(captions)) + " Id max = " + str(objects_ids.max())

            vis = np.ones((len(objects_boxes)))

            ax[0].imshow(self.frames[i].image)

            draw_frame_boxes(self.frames[i].image, visibilities=vis,
                             boxes=objects_boxes, title=title, captions=captions, colors=colors, ax=ax[1])

            plt.show()

    def animate_tracking(self):

        plt.rcParams["animation.html"] = "jshtml"
        fig, ax = get_ax()

        def animate(t):

            ax.clear()

            objects_ids = self.frames[t].get_all_ids()

            objects_boxes = self.frames[t].get_all_boxes()

            captions = list(map(str, objects_ids))

            colors = [self.colors[i] for i in objects_ids]

            vis = 2 * np.ones((len(objects_boxes)))

            title = "Nb. objects = " + \
                str(len(captions)) + " Id max = " + str(objects_ids.max())

            draw_frame_boxes(self.frames[t].image, visibilities=vis, boxes=objects_boxes, captions=captions,
                             title=title, ax=ax, colors=colors)

        anim = animation.FuncAnimation(fig, animate, frames=len(self.frames))

        plt.close()
        return anim

    def animate_gt(self):

        plt.rcParams["animation.html"] = "jshtml"
        fig, ax = get_ax()

        def animate2(t):

            ax.clear()

            objects_ids = self.gt_frames[t].get_all_ids()

            objects_boxes = self.gt_frames[t].get_all_boxes()

            captions = list(map(str, objects_ids))

            colors = [self.colors[i] for i in objects_ids]

            vis = 2 * np.ones((len(objects_boxes)))

            title = "Nb. objects = " + \
                str(len(captions)) + " Id max = " + str(objects_ids.max())

            draw_frame_boxes(self.gt_frames[t].image, visibilities=vis, boxes=objects_boxes, captions=captions,
                             title=title, ax=ax, colors=colors)

        anim = animation.FuncAnimation(
            fig, animate2, frames=len(self.gt_frames))

        plt.close()
        return anim

    def load_gt(self, path_ids="", path_boxes=""):

        files_ids = sorted(os.listdir(path_ids))
        files_boxes = sorted(os.listdir(path_boxes))

        self.gt_frames = np.empty((len(self.frames)), dtype=object)

        for i in range(len(files_boxes)):

            id_frame = skimage.io.imread(
                os.path.join(path_ids, files_ids[i]))

            preds = None if i == 0 else objects_ids.copy()

            objects_ids = np.array(
                [obj.mean_intensity for obj in regionprops(id_frame, id_frame)], np.int32)

            image = skimage.io.imread(
                os.path.join(path_boxes, files_boxes[i]))

            identifier = files_boxes[i]

            image, window, scale, padding, crop = utils.resize_image(
                gray2rgb(image),
                min_dim=self.detection_config.IMAGE_MIN_DIM,
                min_scale=self.detection_config.IMAGE_MIN_SCALE,
                max_dim=self.detection_config.IMAGE_MAX_DIM,
                mode=self.detection_config.IMAGE_RESIZE_MODE)

            mask = utils.resize_mask(self.dataset.load_mask(
                self.dataset.image_ids[i])[0], scale, padding, crop)

            # image_boxes = np.array([obj.bbox for obj in regionprops(image)])

            # image_boxes = rescale_bounding_boxes(
            #     image_boxes, image.shape, self.frames[i].image.shape[:2])

            image_boxes = utils.extract_bboxes(mask)

            # image_boxes=rescale_bounding_boxes(
            #     image_boxes, image.shape, self.frames[i].image.shape[:2])

            self.gt_frames[i] = Frame(
                identifier, self.frames[i].image, image_boxes=image_boxes, idx=i)

            # i set the objects ids and predecessors relationship
            self.gt_frames[i].set_objects_ids(preds, objects_ids)

    def video_tracking(self, interval=500, repeat_delay=10000):

        fig, ax = get_ax()
        frames = []

        for t in range(len(self.frames)):
            # remove the first dim
            ax.clear()

            objects_ids = self.frames[t].get_all_ids()

            objects_boxes = self.frames[t].get_all_boxes()

            captions = list(map(str, objects_ids))

            colors = [self.colors[i] for i in objects_ids]

            vis = 2 * np.ones((len(objects_boxes)))

            title = "Nb. objects = " + \
                str(len(captions)) + " Id max = " + str(objects_ids.max())

            frames.append(
                [draw_frame_boxes(self.frames[t].image, visibilities=vis, boxes=objects_boxes, captions=captions,
                                  title=title, ax=ax, colors=colors, animated=True)])

        anim = animation.ArtistAnimation(fig, frames, interval=interval,
                                         repeat_delay=repeat_delay)

        plt.close()
        # Show the animation
        return HTML(anim.to_html5_video())
