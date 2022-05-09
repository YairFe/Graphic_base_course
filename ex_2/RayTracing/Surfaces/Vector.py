import numpy as np


class Vector:
    def __init__(self, start_point, cross_point):
        self.start_point = start_point
        self.cross_point = cross_point


    def get_reflection_direction(self, normal):
        return self.cross_point - 2 * np.dot(normal, self.cross_point) * normal

    def get_perpendicular_plane(self):
        """
        function which getting right vector and up vector of a perpendicular plane to this
        Return: right direction vector and up direction vector of the perpendicular plane
        """
        # the calculation doesnt work on parallel vectors so in-order to always succeed we test them all
        vectors_list = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        for vector in vectors_list:
            right_direction = np.cross(self.cross_point, vector)
            if np.dot(right_direction, self.cross_point) == 0:
                break
        right_direction /= np.linalg.norm(right_direction)
        up_direction = np.cross(self.cross_point, right_direction)
        return right_direction, up_direction


