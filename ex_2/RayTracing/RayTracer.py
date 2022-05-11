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
        img = np.array(Image.new(mode="RGB", size=(width, height)), dtype=np.float64)
        row_position = self.camera.position + self.camera.distance * self.camera.direction \
                       + (self.camera.screen_width/2) * self.camera.right \
                       + (screen_height/2) * self.camera.up_vector
        for i in range(height):
            pixel_position = np.copy(row_position)
            print(f"{i=}")
            for j in range(width):
                vector = Vector(
                    start_point=np.copy(self.camera.position),
                    cross_point=Vector.normalize_vector(pixel_position-self.camera.position))
                intersection = self.intersect_vector_with_scene(vector)
                if intersection is not None:
                    # intersection 0 - intersection_point
                    # intersection 1 - intersection_normal
                    # intersection 2 - intersection_surface
                    img[i, j] = self.get_color(intersection[0], intersection[1], intersection[2])
                else:
                    img[i, j] = self.background_color
                pixel_position -= self.camera.right * self.camera.screen_width / width
            row_position -= self.camera.up_vector * screen_height / height
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
                pass
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


    def get_color(self, intersection_point=None, normal=None, surface=None):
        """
        calculate the color of a ray at intersection_point with normal provided to the surface from surface_list
        intersection_point(ndArray) : a point in 3D space
        normal(ndArray): direction vector perpendicular to the surface
        surface(Surface): one of the surfaces in surface_list - Cube, Spheres, InfinitePlane
        """
        material = self.material_list[surface.material_index-1]
        intersection_to_camera = Vector(start_point = intersection_point, cross_point = self.camera.position - intersection_point)

        background_color = self.get_background_color()

        total_diffuse_specular_color = np.zeros(3)
        for light in self.light_list:
            intersection_to_light = Vector(start_point=light.position, cross_point=Vector.normalize_vector(intersection_point - light.position))
            light_direction = light.get_normalized_light_direction_vector(intersection_point)
            light_reflection_direction = light.get_normalized_light_reflection_direction_vector(intersection_point, normal)
            diffuse_color = np.dot(normal, light_direction) * material.diffuse_color
            specular_color = (np.dot(light_reflection_direction, intersection_to_camera.cross_point) ** material.phong) * material.specular_color * light.specular_intensity
            light_intensity = self.get_light_intensity(light, intersection_point, intersection_to_light, surface)
            total_diffuse_specular_color += light_intensity * light.color * (diffuse_color + specular_color)
        return np.clip(material.transparency * background_color + (1-material.transparency) * total_diffuse_specular_color, 0, 1)

    def get_background_color(self):
        return self.background_color

    def get_light_intensity(self, light, intersection_point, ray, surface):
        right, up = ray.get_perpendicular_plane()
        cell_size = light.light_radius / self.num_of_shadow_ray
        # calculate most bottom left point in the grid and set it as starting point
        staring_point = light.position - right * light.light_radius / 2 - up * light.light_radius / 2
        # make a grid of NxN contain start position of each bottom left corner in a cell
        grid = np.full((self.num_of_shadow_ray, self.num_of_shadow_ray, 3), staring_point) + \
               np.tile(np.arange(self.num_of_shadow_ray), self.num_of_shadow_ray).reshape((self.num_of_shadow_ray, self.num_of_shadow_ray, 1)) * (right * cell_size) + \
               np.rot90(np.tile(np.arange(self.num_of_shadow_ray), self.num_of_shadow_ray).reshape((self.num_of_shadow_ray, self.num_of_shadow_ray, 1)), -1) * (up * cell_size)
        grid += self._get_random_grid(low=0, high=cell_size)

        def is_intersecting_with_surface(start_point):
            light_ray = Vector(start_point=start_point,cross_point=Vector.normalize_vector(intersection_point - start_point))
            intersection_result = self.intersect_vector_with_scene(light_ray)
            if intersection_result is None:
                return False
            return intersection_result[2] == surface
        num_of_intersecting_rays = np.sum(list(map(lambda x: is_intersecting_with_surface(x), grid.reshape((self.num_of_shadow_ray**2, 3)))))
        percentage_of_intersecting_rays = num_of_intersecting_rays / (self.num_of_shadow_ray ** 2)
        return (1 - light.shadow_intensity) + light.shadow_intensity * percentage_of_intersecting_rays


    def intersect_vector_with_scene(self, vector):
        #returns (P,N,minimum_intersecting_surface) - the first intersection with the scene,
        #or None if the vector doesn't intersect any surface
        min_t = np.inf
        min_normal = None
        minimum_intersecting_surface = None
        for surface in self.surface_list:
            intersecting_surface = surface.intersection_with_vector(vector)
            if intersecting_surface is None:
                continue
            else:
                t, N = intersecting_surface
            if t < min_t:
                min_t = t
                min_normal = N
                minimum_intersecting_surface = surface
        if min_t == np.inf:
            return None
        P = vector.start_point + min_t * vector.cross_point
        return P, min_normal, minimum_intersecting_surface

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

