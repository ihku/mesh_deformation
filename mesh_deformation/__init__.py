#!/usr/bin/env python3

import argparse
import os

from .mesh import read_obj, write_obj
from .rimd import meshes_to_rimds, rimds_to_meshes, write_rimd, read_rimd


def run_all():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser_rimd = subparsers.add_parser('rimd')
    parser_rimd.add_argument('--input', help='path to list_meshes.txt', type=str, required=True)
    parser_rimd.add_argument('--output', help='path to rimd npz output', type=str, required=True)
    parser_mesh = subparsers.add_parser('mesh')
    parser_mesh.add_argument('--input', help='path to rimd npz input', type=str, required=True)
    parser_mesh.add_argument('--mesh0', help='path to base mesh', type=str, required=True)
    parser_mesh.add_argument('--output-dir', help='path to output dir', type=str, required=True)
    args = parser.parse_args()
    if args.command == 'rimd':
        run_meshes2rimd(args.input, args.output)
    elif args.command == 'mesh':
        run_rimd2meshes(args.input, args.mesh0, args.output_dir)


def run_meshes2rimd(input, output):
    meshes = []
    with open(input, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if len(line) > 0:
                meshes.append(read_obj(line))
    rimds = meshes_to_rimds(meshes)
    write_rimd(output, rimds)


def run_rimd2meshes(input, mesh0, output_dir):
    mesh0 = read_obj(mesh0)
    rimds = read_rimd(input)
    meshes = rimds_to_meshes(rimds, mesh0)
    for i, mesh in enumerate(meshes):
        write_obj(os.path.join(output_dir, '%d.obj' % i), mesh)
