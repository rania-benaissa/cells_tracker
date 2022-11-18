from deepcell.datasets.tracked import hela_s3 as hela
from utils import *

filename = 'hela_tracking.trks'
(X_train, y_train), (X_test, y_test) = hela.load_tracked_data(filename)


print("train images", X_train.shape)
print("train labels", y_train.shape)
print("test images", X_test.shape)
print("test labels", y_test.shape)
