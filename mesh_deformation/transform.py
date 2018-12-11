#!/usr/bin/env python3

import numpy as np

from .mesh import Mesh


def apply_transform_to_mesh(mesh: Mesh, transformation: np.ndarray):
    """
    :param mesh: Mesh to be transform
    :param transformation: 3x3 ndarray
    :return: transformed mesh
    """

    vert = mesh.get_vertices()
    ones = np.ones(shape=(vert.shape[0], 1))
    vert = np.concatenate((vert, ones), axis=1)
    vert = vert @ transformation.T
    return Mesh(vert[:, :3] / vert[:, 3, None], mesh.get_polygons())


def eye():
    return np.eye(4, dtype=np.float32)


def scale(value) -> np.ndarray:
    """

    :param value: array of shape 3
    :return: scale matrix
    """
    value = list(value)
    return np.diag(value + [1.0,]).astype(np.float32)
