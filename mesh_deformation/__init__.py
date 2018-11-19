#!/usr/bin/env python3

import sys
import argparse

from .mesh import read_obj
from .rimd import meshes_to_rimds

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
