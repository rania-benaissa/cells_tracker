import numpy as np


class Frame():

    objects_ids = []

    def __init__(self, identifier, image, image_features=None, image_boxes=None):

        self.id = identifier

        self.image = image

        self.objects_features = image_features

        self.objects_boxes = image_boxes

        self.centroids = self.compute_centroids()

    def resize_features(self, features2):

        if len(self.objects_features) < len(features2):

            f1_shape = len(self.objects_features)

            self.objects_features = np.resize(
                self.objects_features, features2.shape)

            for i in range(f1_shape, len(features2)):

                self.objects_features[i] = np.full(features2.shape[-1], 10**5)

    def compute_centroids(self):

        centroids = np.zeros((len(self.objects_boxes), 2))

        for i in range(len(self.objects_boxes)):

            y1, x1, y2, x2 = self.objects_boxes[i]

            centroids[i, 0] = (x1 + x2) / 2
            centroids[i, 1] = (y1 + y2) / 2

        return centroids
