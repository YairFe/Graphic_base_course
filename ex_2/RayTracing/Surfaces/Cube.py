import numpy as np

class Cube:
    def __init__(self, center, edge_length, material_index):
        self.material_index = material_index
        self.edge_length = edge_length
        self.center = center

    def intersection_with_vector(self, vector):
        #returns (t,N) - the minimal t for intersection with the surface and the normal N
        #or None if there is no intersection
        minimal_t = np.inf
        normal = None
        for dim in range(3):
            if vector.cross_point[dim] == 0:
                break
            for sign in [-1,1]:
                t = (self.center[dim] - vector.start_point[dim] + sign*self.edge_length/2)\
                    /vector.cross_point[dim]
                if t<0:
                    break
                P = vector.start_point + t*vector.cross_point
                if P[0] >= self.center[0] - self.edge_length/2 and\
                   P[0] <= self.center[0] + self.edge_length/2 and\
                   P[1] >= self.center[1] - self.edge_length/2 and\
                   P[1] <= self.center[1] + self.edge_length/2 and\
                   P[2] >= self.center[2] - self.edge_length/2 and\
                   P[2] <= self.center[2] + self.edge_length/2 and\
                   t < minimal_t:
                        minimal_t = t
                        normal = np.zeros(3)
                        normal[dim] = sign
        if minimal_t == np.inf:
            return None
        return minimal_t, normal


    def intersection_with_vectors(self, start_points, directions):
        minimal_t = np.full(start_points.shape[0], np.inf)
        normals = np.zeros_like(start_points)
        for dim in range(3):
            denominators = np.where(directions[:,dim] == 0, np.inf, directions[:,dim])
            for sign in [-1,1]:
                t = (self.center[dim] - start_points[:,dim] + sign*self.edge_length/2)/denominators
                P = start_points + t[:,np.newaxis]*directions
                is_minimal = (P[:,0] >= self.center[0] - self.edge_length/2) *\
                             (P[:,0] <= self.center[0] + self.edge_length/2) *\
                             (P[:,1] >= self.center[1] - self.edge_length/2) *\
                             (P[:,1] <= self.center[1] + self.edge_length/2) *\
                             (P[:,2] >= self.center[2] - self.edge_length/2) *\
                             (P[:,2] <= self.center[2] + self.edge_length/2) *\
                             (t > 0) == 1
                minimal_t = np.where(is_minimal, t, minimal_t)
                normals = np.where(np.repeat(is_minimal, 3).reshape(normals.shape), sign * np.array([dim == i for i in range(3)], dtype=np.float64), normals)
        return minimal_t, normals
