#!/usr/bin/env python3

from collections import defaultdict
import numpy as np


class Mesh:
    __slots__ = '_vertices', '_polygons'

    def __init__(self, vertices: np.ndarray, polygons: np.ndarray, copy=False):
        """
        :param vertices: numpy.ndarray of float32, list of points
        :param polygons: numpy.ndarray of int, list of polygons,
                    where each polygon is a list with three vertex indices
        """

        if copy:
            self._vertices = vertices.copy()
            self._polygons = polygons.copy()
        else:
            self._vertices = vertices
            self._polygons = polygons

    def get_vertices(self) -> np.ndarray:
        return self._vertices

    def get_polygons(self) -> np.ndarray:
        return self._polygons

    def get_vertex(self, i: int) -> np.ndarray:
        return self._vertices[i]

    def get_num_vertices(self) -> int:
        return len(self._vertices)

    def get_edges(self) -> dict:
        edges = dict()
        for i, tr in enumerate(self._polygons):
            edges[(tr[0], tr[1])] = i
            edges[(tr[1], tr[0])] = i
            edges[(tr[0], tr[2])] = i
            edges[(tr[2], tr[0])] = i
            edges[(tr[1], tr[2])] = i
            edges[(tr[2], tr[1])] = i
        return edges


def edges_to_adjacency_list(edges) -> dict:
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
    return adj_list


def read_obj(path):
    """
    Implements a simplified Wavefront .obj parser.
    Reads obj at `path`.
    :param path: path to the obj file
    :return: Mesh
    """
    vertices = []
    faces = []
    with open(path, 'r') as fin:
        req_vert = set()
        for line in fin.readlines():
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            tokens = line.split(' ')
            if tokens[0] == 'v':
                vertices.append(list(map(float, tokens[1:])))
            elif tokens[0] == 'f':
                face = [int(t.split('/')[0]) - 1 for t in tokens[1:]]
                req_vert.update(face)
                while len(face) >= 3:
                    faces.append(face[-3:])
                    del face[-2]
    # TODO: some checks
    return Mesh(np.array(vertices, dtype=np.float32),
                np.array(faces, dtype=np.int))


def write_obj(path, mesh, precision=4):
    def rnd(f):
        return round(f, precision)
    with open(path, 'w') as fout:
        for vertice in mesh.get_vertices():
            fout.write('v {} {} {}\n'.format(rnd(vertice[0]), rnd(vertice[1]), rnd(vertice[2])))
        for triangle in mesh.get_polygons():
            fout.write('f {} {} {}\n'.format(triangle[0] + 1, triangle[1] + 1, triangle[2] + 1))
