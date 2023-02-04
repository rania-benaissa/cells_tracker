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


def extract_image_detections(image_id, model, config, dataset, gt_exists=True):

    if gt_exists:

        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id)

        results = model.detect_molded(np.expand_dims(
            image, 0), np.expand_dims(image_meta, 0), verbose=0)

        # Display results
        r = results[0]

        return image, r['rois'], r['masks'], r['scores'], gt_bbox, gt_mask

    else:

        image = dataset.load_image(image_id)

        r = model.detect([image], verbose=0)[0]

        return image, r['rois'], r['masks'], r['scores']


def remove_disappearing_objects(cost_matrix):

    removed_rows = np.where((cost_matrix == 0).all(axis=1))[0]

    # remove those rows from cost matrix
    cost_matrix = np.delete(cost_matrix, removed_rows, axis=0)

    return 1 - cost_matrix, removed_rows


class Tracker():

    def __init__(self, DATASET_DIR2="", weights="", subdir="val", path_ids="", nb_images="all"):

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
        self.fill_sequence(self.dataset, nb_images, path_ids)

        self.colors = random_colors(1000)

        self.max_id = 0

    def fill_sequence(self, dataset, nb_images, path_ids):

        self.frames = np.empty((len(dataset.image_ids)), dtype=object) if(
            nb_images == "all") else np.empty((nb_images), dtype=object)

        self.gt_frames = np.empty((len(self.frames)), dtype=object)

        images_ids = dataset.image_ids if(
            nb_images == "all")else dataset.image_ids[:nb_images]

        if(path_ids != ""):
            files_ids = sorted(os.listdir(path_ids))

        for i in range(len(images_ids)):

            identifier = dataset.image_info[images_ids[i]]["id"]

            if path_ids != "":

                image, image_boxes, image_masks, image_scores, gt_boxes, gt_masks = extract_image_detections(
                    images_ids[i], self.detection_model, self.detection_config, dataset)

                image_masks = np.transpose(image_masks, (2, 0, 1))
                gt_masks = np.transpose(gt_masks, (2, 0, 1))

                self.frames[i] = Frame(
                    identifier, image, image_boxes=image_boxes, image_masks=image_masks, image_scores=image_scores, idx=i)

                self.gt_frames[i] = Frame(
                    identifier, self.frames[i].image, image_boxes=gt_boxes, image_masks=gt_masks, idx=i)

                # this part is to get the groundtruth ids
                id_frame = skimage.io.imread(
                    os.path.join(path_ids, files_ids[i]))

                preds = None if i == 0 else objects_ids.copy()

                objects_ids = np.unique(id_frame)[1:]

                self.gt_frames[i].set_objects_ids(preds, objects_ids)

            else:

                image, image_boxes, image_masks, image_scores = extract_image_detections(
                    images_ids[i], self.detection_model, self.detection_config, dataset, False)  # handle when there's no groundtruth
                image_masks = np.transpose(image_masks, (2, 0, 1))

                self.frames[i] = Frame(
                    identifier, image, image_boxes=image_boxes, image_masks=image_masks, image_scores=image_scores, idx=i)

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


############################################################
#  Evaluation
############################################################

    def compute_MOTA(self, iou_threshold=0.5, score_threshold=0.5):

        # compute matchs per frames
        for i in range(len(self.frames)):

            # all ids are cells ofcs
            pred_class_id = np.ones(len(self.frames[i].objects))
            gt_class_id = np.ones(len(self.gt_frames[i].objects))

            gt_box = self.gt_frames[i].get_all_boxes()

            gt_mask = self.gt_frames[i].get_all_masks()

            pred_box = self.frames[i].get_all_boxes()

            pred_score = self.frames[i].get_all_scores()

            pred_mask = self.frames[i].get_all_masks()

            pred_mask = np.transpose(pred_mask, (1, 2, 0))

            gt_mask = np.transpose(gt_mask, (1, 2, 0))

            gt_match, pred_match, overlaps = utils.compute_matches(
                gt_box, gt_class_id, gt_mask,
                pred_box, pred_class_id, pred_score, pred_mask,
                iou_threshold=iou_threshold, score_threshold=score_threshold)

            for j in range(len(gt_match)):

                if(gt_match[j] != -1 and pred_match[j] != -1):

                    print(
                        "(gt = {} , pred {}, iou {})".format(self.gt_frames[i].objects[int(gt_match[j])].id, self.frames[i].objects[int(pred_match[j])].id, overlaps[j]))

############################################################
#  Visualisation
############################################################

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
