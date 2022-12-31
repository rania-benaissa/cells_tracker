import sys
import os

# remove warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Root directory of the MASK R-CNN Implementation
ROOT_DIR = os.path.abspath("mrcnn/")
sys.path.append(ROOT_DIR)  # To find local version of the library
print("Root directory", ROOT_DIR)


import warnings
warnings.filterwarnings('ignore')

TRAIN_SAVE_DIR = os.path.abspath("train_results")
TEST_SAVE_DIR = os.path.abspath("test_results")

DATASET_DIR = os.path.abspath("Images")

# riri know that your training data is 02


# actually the validation set is  Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
