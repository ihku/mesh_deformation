#!/usr/bin/env python3

import argparse
import os

from .mesh import read_obj, write_obj
from .rimd import meshes_to_rimds, rimds_to_meshes


def run_mesh():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to obj', type=str)
    args = parser.parse_args()

    mesh = read_obj(args.path)
    print(mesh._vertices)


def run_meshes2rimd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to list_meshes.txt', type=str)
    args = parser.parse_args()

    meshes = []
    with open(args.path, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if len(line) > 0:
                meshes.append(read_obj(line))
    print(meshes_to_rimds(meshes))


def test_rimd_meshes():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to list_meshes.txt', type=str)
    parser.add_argument('--output-dir', help='path to output dir', type=str)
    args = parser.parse_args()

    names = []
    meshes = []
    with open(args.path, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if len(line) > 0:
                meshes.append(read_obj(line))
                names.append(os.path.basename(line))
    rimds = meshes_to_rimds(meshes)
    meshes = rimds_to_meshes(rimds, meshes[0])
    for name, mesh in zip(names, meshes):
        write_obj(os.path.join(args.output_dir, name), mesh)
