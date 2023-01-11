
############################################################
#  RLE Encoding
############################################################

import numpy as np
import os
import datetime
from setup import *
import matplotlib.pyplot as plt
from cells_detection.cell import *
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    fig.tight_layout()
    return ax


def prepareDataset(dataset_dir):

    # Validation dataset
    dataset_test = CellDataset()
    dataset_test.load_cell(dataset_dir, "test")
    dataset_test.prepare()

    return dataset_test


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], dtype=bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(
            shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset):
    """Run detection on images in the given directory."""

    # Create directory
    if not os.path.exists(TEST_SAVE_DIR):
        os.makedirs(TEST_SAVE_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(TEST_SAVE_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Load over images
    submission = []
    for image_id in dataset.image_ids:

        plt.figure()
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir,
                    dataset.image_info[image_id]["id"]))


def run_detection(model, dataset, config):

    # Create directory
    if not os.path.exists(TEST_SAVE_DIR):
        os.makedirs(TEST_SAVE_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(TEST_SAVE_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Load over images
    for image_id in dataset.image_ids:

        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(
                dataset, config, image_id)

        info = dataset.image_info[image_id]

        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        # print("Original image shape: ", modellib.parse_image_meta(
        #     image_meta[np.newaxis, ...])["original_image_shape"][0])

        # Run object detection
        results = model.detect_molded(np.expand_dims(
            image, 0), np.expand_dims(image_meta, 0), verbose=1)

        # Display results
        r = results[0]

        # log("gt_class_id", gt_class_id)
        # log("gt_bbox", gt_bbox)
        # log("gt_mask", gt_mask)

        # Compute AP over range 0.5 to 0.95 and print it
        # utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
        #                        r['rois'], r['class_ids'], r['scores'], r['masks'],
        #                        verbose=1)

        visualize.display_differences(
            image,
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            dataset.class_names, ax=get_ax(),
            show_box=True, show_mask=True,
            iou_threshold=0.5, score_threshold=0.5)
        plt.savefig("{}/{}.png".format(submit_dir,
                    dataset.image_info[image_id]["id"]))
