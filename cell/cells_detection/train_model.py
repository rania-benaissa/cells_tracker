############################################################
#  Training
############################################################

from cells_detection.cell import *
from imgaug import augmenters as iaa


def selectWeights(model, weights="coco"):

    # Select weights file to load
    if weights.lower() == "last":

        # Find last trained weights
        weights_path = model.find_last()

    elif weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()

    elif weights.lower() == "coco":
        # Start from coco trained weights
        #weights_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        weights_path = "mask_rcnn_coco.h5"

    elif weights.lower() == "usiigaci":
        # Start from coco trained weights
        #weights_path = os.path.join(ROOT_DIR, "Usiigaci_3.h5")
        weights_path = "Usiigaci_3.h5"

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    return model


def train(model, dataset_dir, config):
    """Train the model."""
    # Training dataset.
    dataset_train = CellDataset()
    dataset_train.load_cell(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellDataset()
    dataset_val.load_cell(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html

    if config.AUGMENT_TRAIN:

        augmentation = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])

    # actually the validation set is
    # Data on which to evaluate the loss and any model metrics at the end of each epoch.
    # The model will not be trained on this data.

    print("Train", config.TRAINABLE_LAYERS, " layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS,
                augmentation=augmentation,
                layers=config.TRAINABLE_LAYERS)
