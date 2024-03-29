import sys
import numpy as np
from PIL import Image

from Camera import Camera
from Light import Light
from Material import Material
from Surfaces.Cube import Cube
from Surfaces.InfinitePlane import InfinitePlane
from Surfaces.Spheres import Sphere
from Surfaces.Vector import Vector


class RayTracer:
    epsilon = 0.0002
    
    def __init__(self,scene_path):
        self.camera, \
        self.background_color, \
        self.num_of_shadow_ray, \
        self.max_recursion, \
        self.material_list, \
        self.light_list, \
        self.surface_list = self.parse_scene(scene_path)

    def render(self, image_name, height, width):
        screen_height = height * self.camera.screen_width / width
        img = np.full((width, height, 3), self.background_color, dtype=np.float64)
        start_points = np.zeros((height, width, 3))
        start_shape = start_points.shape
        row_position = self.camera.position + self.camera.distance * self.camera.direction \
                       + (self.camera.screen_width/2) * self.camera.right \
                       + (screen_height/2) * self.camera.up_vector
        for i in range(height):
            pixel_position = np.copy(row_position)
            for j in range(width):
                start_points[i,j] = pixel_position
                pixel_position -= self.camera.right * self.camera.screen_width / width
            row_position -= self.camera.up_vector * screen_height / height
        directions = start_points - self.camera.position
        directions /= (np.sum(directions**2, axis=2)**0.5)[:,:,np.newaxis]
        intersection = self.intersect_vectors_with_scene(start_points, directions)
        directions = directions.ravel().reshape((directions.size//3, 3))[intersection[3]]
        transparency = np.full(intersection[2].shape, 1, dtype=np.float64)
        reflection = np.full(intersection[0].shape, 1, dtype=np.float64)
        img[intersection[3].reshape(start_shape[:-1])] = self.get_color(intersection[0], intersection[1], intersection[2],\
                                                                        directions, transparency, reflection, 0)
        img = img.reshape(start_shape)
        Image.fromarray((255*img).astype(np.uint8), mode='RGB').save(image_name)


    def parse_scene(self, scene_path):
        with open(scene_path, 'r') as f:
            file_content = [row.split() for row in f.read().splitlines() if not (not row or row.isspace() or row.startswith('#'))]

        material_list = []
        light_list = []
        surface_list = []
        camera = None
        background_color = np.array((0, 0, 0))
        num_of_shadow_ray = 1
        max_recursion = 1
        for row in file_content:
            if row[0] == 'cam':
                camera = Camera(
                    position=np.array((row[1], row[2], row[3]), dtype=np.float64),
                    look_at_point=np.array((row[4], row[5], row[6]), dtype=np.float64),
                    up_vector=np.array((row[7], row[8], row[9]), dtype=np.float64),
                    distance=float(row[10]),
                    screen_width=float(row[11]))
            elif row[0] == 'set':
                background_color = np.array((row[1], row[2], row[3]), dtype=np.float64)
                num_of_shadow_ray = int(row[4])
                max_recursion = int(row[5])
            elif row[0] == 'mtl':
                material_list.append(Material(
                    diffuse_color=np.array((row[1], row[2], row[3]), dtype=np.float64),
                    specular_color=np.array((row[4], row[5], row[6]), dtype=np.float64),
                    reflection_color=np.array((row[7], row[8], row[9]), dtype=np.float64),
                    phong=float(row[10]),
                    transparency=float(row[11]))
                )
            elif row[0] == 'lgt':
                light_list.append(Light(
                    position=np.array((row[1], row[2], row[3]), dtype=np.float64),
                    color=np.array((row[4], row[5], row[6]), dtype=np.float64),
                    specular_intensity=float(row[7]),
                    shadow_intensity=float(row[8]),
                    light_radius=float(row[9])
                    )
                )
            elif row[0] == 'pln':
                surface_list.append(InfinitePlane(
                    normal=np.array((row[1], row[2], row[3]), dtype=np.float64),
                    offset=float(row[4]), material_index=int(row[5])
                ))
            elif row[0] == 'sph':
                surface_list.append(Sphere(
                    center=np.array((row[1], row[2], row[3]), dtype=np.float64),
                    radius=float(row[4]), material_index=int(row[5])
                ))
            elif row[0] == 'box':
                surface_list.append(Cube(
                    center=np.array((row[1], row[2], row[3]), dtype=np.float64),
                    edge_length=float(row[4]), material_index=int(row[5])
                ))
            else:
                raise NotImplemented
        return camera, background_color, num_of_shadow_ray, max_recursion, material_list, light_list, surface_list


    def get_color(self, intersection_points, normals, surfaces, directions, accumulated_transparencies, accumulated_reflection_colors, depth):
        """
        calculate the color of a ray at intersection_point with normal provided to the surface from surface_list
        intersection_point(ndArray) : points on surfaces
        normals(ndArray): direction vectors perpendicular to the surfaces
        surface(ndArray): surfaces in surface_list - Cube, Spheres, InfinitePlane
        valid(ndArray): array of booleans, if False then the appropriate color should be the background color
        """
        material_diffuse_colors = np.ones_like(intersection_points)
        material_specular_colors = np.ones_like(intersection_points)
        material_reflection_colors = np.ones_like(intersection_points)
        material_transparencies = np.ones(intersection_points.shape[0])
        material_phong = np.zeros_like(surfaces, dtype=np.float64)
        for i in range(surfaces.shape[0]):
            material = self.material_list[surfaces[i].material_index-1]
            material_diffuse_colors[i] = material.diffuse_color
            material_specular_colors[i] = material.specular_color
            material_reflection_colors[i] = material.reflection_color
            material_phong[i] = material.phong
            material_transparencies[i] = material.transparency
        accumulated_reflection_colors *= material_reflection_colors
        accumulated_transparencies *= material_transparencies
        if depth < self.max_recursion:
            background_color = self.get_background_color(intersection_points, directions, surfaces,\
                                                         accumulated_transparencies, accumulated_reflection_colors, depth)
        else:
            background_color = np.full_like(intersection_points, self.background_color)
        total_diffuse_specular_colors = np.zeros_like(intersection_points)                                     
        light_intensities = self.get_light_intensities(intersection_points)
        #light_intensities[i,j,k] is the light intensity of light_list[i] at point intersection_point[j,k]
        for i in range(len(self.light_list)):
            light = self.light_list[i]
            light_to_intersections = intersection_points - light.position
            light_to_intersections /= (np.sum(light_to_intersections**2, axis=1)**0.5)[:,np.newaxis]#normalizing
            light_directions = -light_to_intersections
            light_reflection_directions = 2 * np.sum(light_directions*normals, axis=1)[:,np.newaxis]*normals - light_directions
            light_reflection_directions /= (np.sum(light_reflection_directions**2, axis=1)**0.5)[:,np.newaxis] #normalizing
            diffuse_colors = np.sum(normals*light_directions, axis=1)[:,np.newaxis] * material_diffuse_colors
            specular_colors = (np.sum(light_reflection_directions*directions, axis=1) ** material_phong)[:,np.newaxis] * material_specular_colors * light.specular_intensity
            total_diffuse_specular_colors += light_intensities[i,:,np.newaxis] * np.full_like(intersection_points, light.color) * (diffuse_colors + specular_colors)
        if depth < self.max_recursion:
            material_reflection_colors *= self.get_reflection_color(intersection_points, directions, normals,\
                                                         accumulated_transparencies, accumulated_reflection_colors, depth)
        else:
            material_reflection_colors = np.zeros_like(intersection_points)
        return np.clip(material_reflection_colors + material_transparencies[:,np.newaxis] * background_color + (1-material_transparencies)[:,np.newaxis] * total_diffuse_specular_colors, 0, 1)


    def get_light_intensities(self, intersection_points):
        grids = self.get_grids(intersection_points)
        points = np.tile(intersection_points, (1,1,self.num_of_shadow_ray ** 2)).reshape((grids[0].shape))
        points = np.array([np.copy(points) for _ in self.light_list])
        directions = grids - points
        directions /= (np.sum(directions**2, axis = 3)**0.5)[:,:,:,np.newaxis]
        points += RayTracer.epsilon * directions
        intersections = self.intersect_vectors_with_scene(points, directions)
        num_of_intersecting_rays = np.sum(1 - intersections[3].reshape((grids.shape[:-1])), axis=2)
        percentage_of_intersecting_rays = num_of_intersecting_rays / (self.num_of_shadow_ray ** 2)
        return np.array([(1 - self.light_list[i].shadow_intensity) + self.light_list[i].shadow_intensity * percentage_of_intersecting_rays[i] for i in range(len(self.light_list))])


    def get_grids(self, intersection_points):
        lights = self.light_list
        n = len(lights)
        directions = np.array([intersection_points - lights[i].position for i in range(n)])
        right = np.zeros((n, intersection_points.shape[0], 3))
        
        for vector in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
            temp = np.cross(vector, directions)
            right = np.where(np.repeat(np.sum(temp**2, axis=2) == 0, 3).reshape(right.shape), right, temp)
        right /= (np.sum(right**2, axis=2)**0.5)[:,:,np.newaxis]
        up = np.cross(directions, right)
        up /= (np.sum(up**2, axis=2)**0.5)[:,:,np.newaxis]
        
        starting_points = np.array([lights[i].position - (right[i]+up[i]) * lights[i].light_radius/2 for i in range(n)])
        cell_size = [light.light_radius / self.num_of_shadow_ray for light in lights]
        grids = np.zeros((n, intersection_points.shape[0], self.num_of_shadow_ray ** 2, 3))
        for i in range(n):
            right_resized = np.repeat(right[i] * cell_size[i], self.num_of_shadow_ray ** 2, axis=0)\
                            .reshape((intersection_points.shape[0], self.num_of_shadow_ray**2, 3))
            up_resized = np.repeat(up[i] * cell_size[i], self.num_of_shadow_ray ** 2, axis=0)\
                         .reshape((intersection_points.shape[0], self.num_of_shadow_ray**2, 3))
            
            grids[i, :] = np.tile(starting_points[i], (1,1,self.num_of_shadow_ray ** 2)).reshape((grids[i].shape))
            grids[i] += np.tile(np.arange(self.num_of_shadow_ray), (intersection_points.shape[0], self.num_of_shadow_ray))\
                    .reshape((intersection_points.shape[0], self.num_of_shadow_ray**2, 1)) * right_resized
            grids[i] += np.tile(np.arange(self.num_of_shadow_ray)[:,np.newaxis], (intersection_points.shape[0], self.num_of_shadow_ray))\
                    .reshape((intersection_points.shape[0] ,self.num_of_shadow_ray**2, 1)) * up_resized
            
            random_grid = np.random.uniform(low=0, high=cell_size[i], size=(intersection_points.shape[0], self.num_of_shadow_ray, self.num_of_shadow_ray, 3))
            random_grid[:, :, :, 0] = 0
            grids[i] += random_grid.reshape(grids[i].shape)
        return grids.reshape((n, intersection_points.shape[0], self.num_of_shadow_ray**2, 3))


    def get_reflection_color(self, intersection_points, directions, normals, transparencies, reflections, depth):
        print(f"reflection {depth}")
        valid = np.linalg.norm(reflections, axis=1) > 0.01
        if np.sum(valid) == 0:
            return np.zeros_like(reflections)
        reflection_direction = directions - 2 * np.sum(normals * directions, axis=1).reshape((directions.shape[0],1)) * normals
        reflection_direction /= (np.sum(reflection_direction**2, axis=1)**0.5)[:,np.newaxis]
        starting_point = intersection_points + reflection_direction * RayTracer.epsilon
        intersection = self.intersect_vectors_with_scene(starting_point[valid], reflection_direction[valid])
        color_array = self.get_color(intersection[0], intersection[1], intersection[2], reflection_direction[valid][intersection[3]],\
                                     transparencies[valid][intersection[3]], reflections[valid][intersection[3]], depth+1)
        result = np.full_like(reflections, self.background_color)
        result[np.flatnonzero(valid)[intersection[3]]] = color_array
        return result

    def get_background_color(self, starting_points, directions, surfaces, transparencies, reflections, depth):
        print(f"background {depth}")
        background_color_array = np.full_like(starting_points, self.background_color, dtype=np.float64)
        valid = transparencies > 0.01
        if np.sum(valid) == 0:
            return background_color_array
        intersection = self.intersect_vectors_with_scene(starting_points[valid], directions[valid], surfaces[valid])
        color_array = self.get_color(intersection[0], intersection[1], intersection[2], directions[valid][intersection[3]],\
                                     np.copy(transparencies[valid][intersection[3]]), np.copy(reflections[valid][intersection[3]]),depth+1)
        background_color_array[np.flatnonzero(valid)[intersection[3]]] = color_array
        return background_color_array

    def intersect_vectors_with_scene(self, start_points, directions, current_surface=None):
        start_points = start_points.ravel().reshape((start_points.size//3, 3))
        directions = directions.ravel().reshape((directions.size//3, 3))
        min_t = np.full(start_points.shape[0], np.inf)
        min_normal = np.zeros_like(start_points)
        minimum_intersecting_surface = np.full_like(min_t, None)
        for surface in self.surface_list:
            t,N = surface.intersection_with_vectors(start_points, directions)
            is_intersecting = (t < min_t) * (t > 0) * ((current_surface is None) + (current_surface != surface))
            min_t = np.where(is_intersecting, t, min_t)
            min_normal = np.where(np.repeat(is_intersecting, 3).reshape(min_normal.shape), N, min_normal)
            minimum_intersecting_surface = np.where(is_intersecting, surface, minimum_intersecting_surface)
        P = start_points + min_t[:,np.newaxis] * directions
        valid = (min_t < np.inf)
        return P[valid], min_normal[valid], minimum_intersecting_surface[valid], valid

    def _get_random_grid(self, low, high):
        random_grid = np.random.uniform(low=low, high=high, size=(self.num_of_shadow_ray, self.num_of_shadow_ray, 3))
        random_grid[:, :, 0] = 0
        return random_grid


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(f"Usage: {__name__} full_path_to_scene image_name <width> <height>")
        exit(1)
    height = 500
    width = 500
    if len(sys.argv) > 4:
        height = int(sys.argv[4])
    if len(sys.argv) > 3:
        width = int(sys.argv[3])
    scene_path = sys.argv[1]
    image_name = sys.argv[2]
    RT = RayTracer(scene_path)
    RT.render(image_name, height, width)

