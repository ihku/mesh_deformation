#!/usr/bin/env python3

import argparse
import os

import numpy as np

from .mesh import read_obj, write_obj
from .rimd import meshes_to_rimds, rimds_to_meshes, write_rimd, read_rimd
from .utils import savetxt

def run_all():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    parser_rimd = subparsers.add_parser('rimd', help='Convert meshes to RIMD')
    parser_rimd.add_argument('--input', help='path to list_meshes.txt', type=str, required=True)
    parser_rimd.add_argument('--output', help='path to rimd npz output', type=str, required=True)
    parser_rimd.add_argument('--output-rs', help='path to rs output at txt', type=str, default=None)
    parser_rimd.add_argument('--output-ss', help='path to ss output at txt', type=str, default=None)
    parser_mesh = subparsers.add_parser('mesh', help='Convert RIMD to meshes')
    parser_mesh.add_argument('--input', help='path to rimd npz input', type=str, required=True)
    parser_mesh.add_argument('--mesh0', help='path to base mesh', type=str, required=True)
    parser_mesh.add_argument('--output-dir', help='path to output dir', type=str, required=True)
    parser_mesh = subparsers.add_parser('check', help='Check RIMDs on some input')
    parser_mesh.add_argument('--input', help='path to rimd npz input', type=str, required=True)
    parser_mesh.add_argument('--output', help='path to output dir', type=str, default=None)
    args = parser.parse_args()
    if args.command == 'rimd':
        run_meshes2rimd(args.input, args.output, args.output_rs, args.output_ss)
    elif args.command == 'mesh':
        run_rimd2meshes(args.input, args.mesh0, args.output_dir)
    elif args.command == 'check':
        check_meshes_and_rs(args.input)
    else:
        raise Exception('unsupported command: ', args.command)


def parse_meshes_file(input):
    meshes = []
    with open(input, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if len(line) > 0:
                meshes.append(read_obj(line))
    return meshes


def write_meshes_to_dir(dirname, meshes):
    for i, mesh in enumerate(meshes):
        write_obj(os.path.join(dirname, '%d.obj' % i), mesh)


def check_meshes_and_rs(input, output):
    meshes = parse_meshes_file(input)
    rimds, rs, *_ = meshes_to_rimds(meshes, output_rs=True)
    meshes_new, rs_new = rimds_to_meshes(rimds, meshes[0], output_rs=True, rs_help=rs)
    assert np.allclose(rs, rs_new)


def run_meshes2rimd(input, output, output_rs=None, output_ss=None):
    if output_rs == output_ss and output_rs is not None:
        raise Exception('output_ss should be different form output_rs')
    meshes = parse_meshes_file(input)
    kwargs = {
        'output_rs': output_rs is not None,
        'output_ss': output_ss is not None,
    }
    result = meshes_to_rimds(meshes, **kwargs)
    write_rimd(output, result.rimd)
    if output_rs is not None:
        savetxt(output_rs, result.rs, (']]]\n\n\n[[[', ']],\n\n[[', '],\n[', ', '), '[[[', ']]]')
    if output_ss is not None:
        savetxt(output_ss, result.ss, (']]]\n\n\n[[[', ']],\n\n[[', '],\n[', ', '), '[[[', ']]]')


def run_rimd2meshes(input, mesh0, output_dir):
    mesh0 = read_obj(mesh0)
    rimds = read_rimd(input)
    meshes = rimds_to_meshes(rimds, mesh0)
    write_meshes_to_dir(output_dir, meshes.meshes)
