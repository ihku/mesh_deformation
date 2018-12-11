#!/usr/bin/env python3

import sys
import os

from mesh_deformation.mesh import read_obj, write_obj
from mesh_deformation.transform import apply_transform_to_mesh, scale
from mesh_deformation.rimd import rimds_to_meshes, meshes_to_rimds
from mesh_deformation.render import render

TMP_DIR = 'TMP'
OUT_DIR = 'OUT'

prepared = False


def prepare():
    global prepared
    if not prepared:
        if not os.path.isdir(TMP_DIR):
            os.mkdir(TMP_DIR)
        if not os.path.isdir(OUT_DIR):
            os.mkdir(OUT_DIR)
        prepared = True


def write_meshes(meshes, path, prefix, suffix):
    for i, mesh in enumerate(meshes):
        write_obj(os.path.join(path, prefix + str(i) + suffix), mesh)


def do_spheres(flags):
    prepare()
    sphere0 = read_obj('meshes/examples/sphere.obj')
    spheres = [sphere0] + [
        apply_transform_to_mesh(sphere0, scale([1, 1, 1 + i / 10])) for i in range(1, 7)
    ]
    rimds = meshes_to_rimds(spheres)
    spheres_new = rimds_to_meshes(rimds.rimd, spheres[0])
    write_meshes(spheres, OUT_DIR, 'sphere_', '.obj')
    write_meshes(spheres_new.meshes, OUT_DIR, 'sphere_new_', '.obj')
    n_meshes = len(spheres)
    if 'render' in flags:
        for i in range(n_meshes):
            render(os.path.join(OUT_DIR, 'sphere_{}.obj'.format(str(i))),
                   os.path.join(OUT_DIR, 'sphere_{}.png'.format(str(i))))
            render(os.path.join(OUT_DIR, 'sphere_new_{}.obj'.format(str(i))),
                   os.path.join(OUT_DIR, 'sphere_new_{}.png'.format(str(i))))


if __name__ == '__main__':
    args = set(sys.argv[1:])
    COMMANDS_LIST = ['spheres']
    FLAGS_LIST = ['render']
    commands = []
    flags = []
    for arg in args:
        if arg in COMMANDS_LIST:
            commands.append(arg)
        elif arg in FLAGS_LIST:
            flags.append(arg)
        else:
            raise Exception('Unexpected argument: {}'.format(arg))

    for command in commands:
        if command == 'spheres':
            do_spheres(flags)
