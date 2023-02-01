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

        self.SEQUENCE_DIR = "Images" if DATASET_DIR2 == "" else DATASET_DIR2

        self.detection_weights = "train_results3/cell20230123T1638/mask_rcnn_cell_0080.h5" if weights == "" else weights

        print("Weights: ", self.detection_weights)
        print("Dataset: ", self.SEQUENCE_DIR)

        # # Configurations

        self.detection_config = CellInferenceConfig()

        dataset = prepareDataset(self.SEQUENCE_DIR, subdir)

        # # Create model
        self.detection_model = modellib.MaskRCNN(mode="inference", config=self.detection_config,
                                                 model_dir=TEST_SAVE_DIR)

        # select weights
        self.detection_model = selectWeights(
            self.detection_model, self.detection_weights)

        self.fill_sequence(dataset, nb_images)
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
                identifier, image, image_boxes=image_boxes)

    def get_ids(self, size, id_image1, row_ind, col_ind):

        id_image2 = -1 * np.ones(size, dtype=np.int16)

        id_image_prec = np.delete(id_image1, self.removed_obj, axis=0)

        # it will have min match anyways
        for i in range(len(col_ind)):

            id_image2[col_ind[i]] = id_image_prec[row_ind[i]]

        self.max_id = self.max_id if self.max_id >= id_image2.max() else id_image2.max()

        for i in range(len(id_image2)):

            if id_image2[i] == -1:

                id_image2[i] = self.max_id + 1
                self.max_id += 1

        return id_image2

    def track_objects(self):

        # for first frame
        self.frames[0].objects_ids = np.arange(
            1, len(self.frames[0].objects_boxes) + 1)  # id_image1

        # for second frame
        cost_matrix, self.removed_obj = self.cost_matrix(
            self.frames[0], self.frames[1])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # get the max id already given
        self.max_id = self.frames[0].objects_ids.max()

        size_id_image2 = len(self.frames[1].objects_boxes)

        self.frames[1].objects_ids = self.get_ids(
            size_id_image2, self.frames[0].objects_ids, row_ind, col_ind)

        for i in range(2, len(self.frames)):

            cost_matrix, self.removed_obj = self.cost_matrix(
                self.frames[i - 1], self.frames[i])

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            size_id_image2 = len(self.frames[i].objects_boxes)

            self.frames[i].objects_ids = self.get_ids(size_id_image2,
                                                      self.frames[i - 1].objects_ids, row_ind, col_ind)

        self.sort_tracking()

    def sort_tracking(self):

        for frame in self.frames:

            frame.objects_boxes = np.array(
                [x for _, x in sorted(zip(frame.objects_ids, frame.objects_boxes))])

            frame.centroids = np.array(
                [x for _, x in sorted(zip(frame.objects_ids, frame.centroids))])

            frame.objects_ids = np.array(sorted(frame.objects_ids))

    def cost_matrix(self, frame1, frame2):

        # add positional information (positional features)

        iou_cost_matrix = compute_overlaps(
            frame1.objects_boxes, frame2.objects_boxes)

        return remove_disappearing_objects(iou_cost_matrix)

    def show_tracking(self):

        for i in range(len(self.frames)):

            fig, ax = get_ax(cols=2)

            captions = list(map(str, self.frames[i].objects_ids))

            colors = [self.colors[i] for i in self.frames[i].objects_ids]

            title = "Nb. objects = " + \
                str(len(captions)) + " Id max = " + \
                str(self.frames[i].objects_ids.max())

            vis = np.ones((len(self.frames[i].objects_boxes)))

            ax[0].imshow(self.frames[i].image)

            draw_frame_boxes(self.frames[i].image, visibilities=vis,
                             boxes=self.frames[i].objects_boxes, title=title, captions=captions, colors=colors, ax=ax[1])

            plt.show()

    def animate_tracking(self):

        plt.rcParams["animation.html"] = "jshtml"
        fig, ax = get_ax()

        def animate(t):

            ax.clear()
            captions = list(map(str, self.frames[t].objects_ids))

            colors = [self.colors[i] for i in self.frames[t].objects_ids]

            vis = 2 * np.ones((len(self.frames[t].objects_boxes)))

            title = "Nb. objects = " + \
                str(len(captions)) + " Id max = " + \
                str(self.frames[t].objects_ids.max())

            draw_frame_boxes(self.frames[t].image, visibilities=vis, boxes=self.frames[t].objects_boxes, captions=captions,
                             title=title, ax=ax, colors=colors)

        anim = animation.FuncAnimation(fig, animate, frames=len(self.frames))

        plt.close()
        return anim
