import re
import sys
import numpy as np
from PIL import Image

from ex_2.RayTracing.Camera import Camera
from ex_2.RayTracing.Light import Light
from ex_2.RayTracing.Material import Material
from ex_2.RayTracing.Surfaces.Cube import Cube
from ex_2.RayTracing.Surfaces.InfinitePlane import InfinitePlane
from ex_2.RayTracing.Surfaces.Spheres import Sphere


def main(scene_path, image_name, width, height):
    camera,\
    background_color,\
    num_of_shadow_ray,\
    max_recursion,\
    material_list,\
    light_list,\
    surface_list = parse_scene(scene_path)
    img = np.array(Image.new(mode="RGB", size=(width, height)))
    row_position = camera.position + camera.distance * camera.direction\
                     + (width/2) * camera.right - (heigth/2) * camera.up_vector
    for i in range(height):
        pixel_position = np.copy(row_position)
        for j in range(width):
            vector = Vector(
                start_point = np.copy(camera.position),
                cross_point = pixel_position-camera.position)
            P, N = intersect_vactor_with_scene(vector, surface_list)
            #TO DO: Implement:
            #img[i,j] = get_color(P, N)
            pixel_position -= camera.right
        row_position += camera.up_vector
    


def parse_scene(scene_path):
    with open(scene_path, 'r') as f:
        file_content = [row.split() for row in f.read().splitlines() if not (row.isspace() or row.startswith('#'))]
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
            background_color = np.array((row[1], row[2], row[3]), dtype=np.int16)
            num_of_shadow_ray = row[4]
            max_recursion = row[5]
        elif row[0] == 'mtl':
            material_list.append(Material(
                diffuse_color=np.array((row[1], row[2], row[3]), dtype=np.int16),
                specular_color=np.array((row[4], row[5], row[6]), dtype=np.int16),
                reflection_color=np.array((row[7], row[8], row[9]), dtype=np.int16),
                phong=float(row[10]),
                transparency=float(row[11]))
            )
        elif row[0] == 'lgt':
            light_list.append(Light(
                position=np.array((row[1], row[2], row[3]), dtype=np.float64),
                color=np.array((row[4], row[5], row[6]), dtype=np.int16),
                specular_intensity=float(row[7]),
                shadow_intensity=float(row[8]),
                light_radius=float(row[9])
                )
            )
        elif row[0] == 'pln':
            surface_list.append(InfinitePlane(
                normal=np.array((row[1], row[2], row[3]), dtype=np.float64),
                offset=float(row[4])
            ))
        elif row[0] == 'sph':
            surface_list.append(Sphere(
                center=np.array((row[1], row[2], row[3]), dtype=np.float64),
                radius=float(row[4])
            ))
        elif row[0] == 'box':
            surface_list.append(Cube(
                center=np.array((row[1], row[2], row[3]), dtype=np.float64),
                edge_length=float(row[4])
            ))
        else:
            raise NotImplemented
        return camera, background_color, num_of_shadow_ray, max_recursion, material_list, light_list, surface_list


def intersect_vactor_with_scene(vector, surface_list):
    #returns (P,N) - the first intersection with the scene,
    #or None if the vector doesn't intersect any surface
    min_t = np.inf
    min_normal = None
    for surface in surface_list:
        t,N = surface.intersection_with_vector(vector)
        if t < min_t:
            min_t = t
            min_normal = N
    if min_t == np.inf:
        return None
    P = vector.start_point + min_t * vecto.cross_point
    return P, min_normal


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(f"Usage: {__name__} full_path_to_scene image_name <width> <height>")
        exit(1)
    height = 500
    width = 500
    if len(sys.argv) > 4:
        height = sys.argv[4]
    if len(sys.argv) > 3:
        width = sys.argv[3]
    scene_path = sys.argv[1]
    image_name = sys.argv[2]
    main(scene_path, image_name, width, height)
