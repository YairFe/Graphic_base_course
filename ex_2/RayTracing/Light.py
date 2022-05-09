import numpy as np


class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, light_radius):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.light_radius = light_radius

    def get_normalized_light_direction_vector(self, start_position):
        temp = self.position - start_position
        return temp / np.linalg.norm(temp)

    def get_normalized_light_reflection_direction_vector(self, start_position, normal):
        light_direction = self.get_normalized_light_direction_vector(start_position)
        temp = 2 * np.dot(light_direction, normal) * normal - light_direction
        return temp / np.linalg.norm(temp)



