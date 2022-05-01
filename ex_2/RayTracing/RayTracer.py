import re
import sys
import numpy as np

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


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(f"Usage: {__name__} full_path_to_scene image_name <width> <height>")
    height = 500
    width = 500
    if len(sys.argv) > 4:
        height = sys.argv[4]
    if len(sys.argv) > 3:
        width = sys.argv[3]
    scene_path = sys.argv[1]
    image_name = sys.argv[2]
    main(scene_path, image_name, width, height)