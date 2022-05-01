import numpy as np


class InfinitePlane:
    def __init__(self, normal, offset):
        self.normal = normal
        self.offset = offset

    def is_intersecting_with_vector(self, vector):
        raise NotImplemented
