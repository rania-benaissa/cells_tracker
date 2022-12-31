############################################################
#  Dataset
############################################################
import os
import numpy as np
import skimage.io

from mrcnn import utils


class CellDataset(utils.Dataset):

    def load_cell(self, dataset_dir, subset):
        """Load a subset of the cell dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset cell, and the class cell
        self.add_class("cell", 1, "cell")

        self.subset = subset

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test"]

        dataset_dir = os.path.join(dataset_dir, subset)

        # Get image ids from directory names
        image_ids = [filename.replace('.tif', '')
                     for filename in os.listdir(dataset_dir)]

        # Add images
        for image_id in image_ids:
            self.add_image(
                "cell",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id + ".tif".format(image_id)))

    # my own function to extract masks
    def extractMasks(self, mask_image):

        masks = []
        # ignore background
        unique = np.unique(mask_image)[1:]

        for value in unique:

            mask = np.where(mask_image != value, 0, 255)

            masks.append(mask.astype(bool))

        return masks

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]

        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(
            os.path.dirname(info['path'])), "GT_" + self.subset)

        # Read mask files from .png image
        mask = []
        filename = "man_seg" + info['id'][1:] + ".tif"

        m = skimage.io.imread(os.path.join(
            mask_dir, filename))

        mask = self.extractMasks(m)

        mask = np.stack(mask, axis=-1)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
