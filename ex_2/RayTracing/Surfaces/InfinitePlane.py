import numpy as np


class InfinitePlane:
    def __init__(self, normal, offset):
        self.normal = normal
        self.offset = offset

    def intersection_with_vector(self, vector):
        #returns (t,N) - the minimal t for intersection with the surface and the normal N
        #or None if there is no intersection
        denominator = np.dot(vector.cross_point, self.normal)
        if denominator == 0:
            return None
        t = (self.offset - np.dot(vector.start_point, self.normal)) / denominator
        if denominator < 0:
            return t, np.copy(self.normal)
        if denominator > 0:
            return t, -self.normal
