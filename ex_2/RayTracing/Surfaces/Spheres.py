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


    def intersection_with_vectors(self, start_points, directions):
        is_in = np.sum((start_points - self.center)**2, axis=1) < self.radius**2
        
        L = self.center - start_points
        t_ca = np.sum(L * directions, axis=1)
        d = np.sum(L*L, axis=1) - t_ca**2
        t = np.where(self.radius**2 >= d ,(self.radius**2 - d), np.inf).astype(np.float64)
        t = np.where(is_in, t_ca + np.sqrt(t), np.where((t < np.inf) * (t_ca >= 0), t_ca - np.sqrt(t), 0))
        P = start_points + t[:,np.newaxis]*directions
        N = (P-self.center) / (np.sum((P-self.center)**2, axis=1)**0.5)[:,np.newaxis]
        return t, N
    
