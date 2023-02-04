import numpy as np


class Object():

    def __init__(self, identifier, bbox, mask, score):

        self.bbox = bbox

        self.isNew = True

        self.mask = mask

        self.score = score

        self.id = identifier

        self.centroid = self.compute_centroid()

    def compute_centroid(self):

        centroid = np.zeros(2)

        y1, x1, y2, x2 = self.bbox

        centroid[0] = (x1 + x2) / 2
        centroid[1] = (y1 + y2) / 2

        return centroid
