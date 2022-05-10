import numpy as np

class Sphere:
    def __init__(self, center, radius, material_index):
        self.material_index = material_index
        self.center = center
        self.radius = radius

    def intersection_with_vector(self, vector):
        #returns (t,N) - the minimal t for intersection with the surface and the normal N
        #or None if there is no intersection
        L = self.center - vector.start_point
        t_ca = np.dot(L, vector.cross_point)
        if t_ca < 0:
            return None
        d = np.dot(L, L) - t_ca**2
        if d > self.radius**2:
            return None
        t = t_ca - (self.radius**2 - d)**0.5
        P = vector.start_point + t*vector.cross_point
        N = (P-self.center) / np.linalg.norm(P-self.center)
        return t, N

    
