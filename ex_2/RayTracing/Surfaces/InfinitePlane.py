import numpy as np


class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.material_index = material_index
        self.normal = normal
        self.offset = offset

    def intersection_with_vector(self, vector):
        #returns (t,N) - the minimal t for intersection with the surface and the normal N
        #or None if there is no intersection
        denominator = np.dot(vector.cross_point, self.normal)
        if denominator == 0:
            return None
        t = (self.offset - np.dot(vector.start_point, self.normal)) / denominator
        if t < 0:
            return None
        return t, self.normal

    def intersection_with_vectors(self, start_points, directions):
        denominators = np.sum(directions*self.normal, axis=1)
        denominators = np.where(denominators == 0, np.inf, denominators)
        t = (self.offset - np.sum(start_points*self.normal, axis=1)) / denominators
        N = np.full_like(start_points, self.normal)
        return t,N
