import numpy as np


class Vector:
    epsilon = 0.0001
    
    def __init__(self, start_point, cross_point, offset=False):
        self.start_point = start_point
        self.cross_point = Vector.normalize_vector(cross_point)
        if offset:
            self.start_point += Vector.epsilon * self.cross_point
        

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
            right_direction = np.cross(vector, self.cross_point)
            if np.dot(right_direction, self.cross_point) == 0:
                break
        right_direction /= np.linalg.norm(right_direction)
        up_direction = np.cross(self.cross_point, right_direction)
        up_direction /= np.linalg.norm(up_direction)
        return right_direction, up_direction

    @staticmethod
    def normalize_vector(point):
        return point / np.linalg.norm(point)
