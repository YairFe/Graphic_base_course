

class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, phong, transparency):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.reflection_color = reflection_color
        self.phong = phong
        self.transparency = transparency