import numpy as np


class Camera:
    def __init__(self, position, look_at_point, up_vector, distance, screen_width):
        self.position = position
        self.look_at_point = look_at_point
        self.distance = distance
        self.screen_width = screen_width
        self.direction = self._get_direction()
        self.up_vector = self._get_up_vector(up_vector, self.direction)
        self.right = np.cross(self.up_vector, self.direction)

    def _get_direction(self, position, look_at_point):
        temp = position - look_at_point
        return temp / np.linalg.norm(temp)

    def _get_up_vector(self, up_vector, direction):
        temp = np.cross(up_vector, direction)
        return temp / np.linalg.norm(temp)

