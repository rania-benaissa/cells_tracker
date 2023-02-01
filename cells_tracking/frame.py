import numpy as np
from cells_tracking.object import *


class Frame():

    def __init__(self, identifier, image, image_features=None, image_boxes=None, idx=-1):

        self.id = identifier

        self.idx = idx

        self.image = image

        self.create_objects(
            image_boxes, image_features)

    def create_objects(self, image_boxes, image_features=None):

        self.objects = np.empty((len(image_boxes)), dtype=object)

        for i in range(len(image_boxes)):

            identifier = i + 1 if self.idx == 0 else -1

            self.objects[i] = Object(identifier, image_boxes[i], features=None)

    def get_all_ids(self):

        ids = np.zeros((len(self.objects)), np.int32)

        for i, obj in enumerate(self.objects):

            ids[i] = obj.id
        return ids

    def get_all_boxes(self):

        boxes = np.zeros((len(self.objects), 4), np.int32)

        for i, obj in enumerate(self.objects):

            boxes[i] = obj.bbox
        return boxes

    def get_known_objects(self, indices):

        centroids = np.zeros((len(indices), 2), np.int32)

        ids = np.zeros(len(indices), np.int32)

        for idx, i in enumerate(indices):

            centroids[idx] = self.objects[i].centroid

            ids[idx] = self.objects[i].id
        return centroids, ids

    def get_unknown_objects(self):

        centroids = []
        pos = []

        for i, obj in enumerate(self.objects):
            if obj.id == -1:

                centroids.append(obj.centroid)
                pos.append(i)

        return np.array(centroids), np.array(pos)

    def get_max_id(self):

        max_id = -1

        for obj in self.objects:

            if(obj.id > max_id):

                max_id = obj.id
        return max_id

        # def resize_features(self, features2):

        #     if len(self.objects_features) < len(features2):

        #         f1_shape = len(self.objects_features)

        #         self.objects_features = np.resize(
        #             self.objects_features, features2.shape)

        #         for i in range(f1_shape, len(features2)):

        #             self.objects_features[i] = np.full(features2.shape[-1], 10**5)
