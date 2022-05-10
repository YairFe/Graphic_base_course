import numpy as np


class Camera:
    def __init__(self, position, look_at_point, up_vector, distance, screen_width):
        self.position = position
        self.look_at_point = look_at_point
        self.distance = distance
        self.screen_width = screen_width
        self.direction = self._get_direction(self.position, self.look_at_point)
        self.right = self._get_right_vector(up_vector, self.direction)
        self.up_vector = np.cross(self.direction, self.right)

    def _get_direction(self, position, look_at_point):
        temp = look_at_point - position
        return temp / np.linalg.norm(temp)

    def _get_right_vector(self, up_vector, direction):
        temp = np.cross(up_vector, direction)
        return temp / np.linalg.norm(temp)

