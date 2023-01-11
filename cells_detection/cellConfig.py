############################################################
#  Configurations
############################################################


import numpy as np
import os
import sys

from mrcnn.config import Config

from setup import *


class CellConfig(Config):

    """Configuration for training on the cell segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "cell"

    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cell

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between cell and BG
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels (needs to be small )
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # possibility to change that

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9  # 0.5

    # How many anchors per image to use for RPN training
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128  # 256  # 128

    # Image mean (RGB)
    #MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    # computed :
    MEAN_PIXEL = np.array([10.01, 10.01, 10.01])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400  # i have more than 300 cells per image so...

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    # PARAMETERS I ADDED

    # i added epochs
    EPOCHS = 1
    # it can be  heads: The RPN, classifier and mask heads of the network
    # all: All the layers
    #     3 + : Train Resnet stage 3 and up
    #     4 + : Train Resnet stage 4 and up
    #     5 + : Train Resnet stage 5 and up
    TRAINABLE_LAYERS = "all"

    AUGMENT_TRAIN = True

    def __init__(self, epochs=5, is_aug=True, bsize=6, lr=10e-3):
        super(CellConfig, self).__init__()

        self.IMAGES_PER_GPU = bsize  # basically it was six

        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Number of training and validation steps per epoch
        # aka nb iterations per epoch
        self.STEPS_PER_EPOCH = len(os.listdir(os.path.join(
            DATASET_DIR, "train"))) // self.IMAGES_PER_GPU

        self.VALIDATION_STEPS = max(
            1, len(os.listdir(os.path.join(DATASET_DIR, "val"))) // self.IMAGES_PER_GPU)

        # i added epochs
        self.EPOCHS = epochs
        self.AUGMENT_TRAIN = is_aug
        self.LEARNING_RATE = lr


class CellInferenceConfig(CellConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    def __init__(self, bsize=1):
        super(CellConfig, self).__init__()

        self.IMAGES_PER_GPU = bsize  # basically it was six

        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        self.IMAGE_RESIZE_MODE = "pad64"
        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        self.RPN_NMS_THRESHOLD = 0.7

        self.USE_MINI_MASK = False
