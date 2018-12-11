#!/usr/bin/env python3

import numpy as np


EPS = 1e-6

def sqrtm_db(A: np.ndarray) -> np.ndarray:
    # denman & beavers, 1976
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    X = A
    Y = np.eye(A.shape[0])
    while np.abs(X * X - A).max() > EPS:
        iX = np.linalg.inv(X)
        iY = np.linalg.inv(Y)
        X = (X + iY) / 2
        Y = (Y + iX) / 2
    return X


def logm_custom(A: np.ndarray) -> np.ndarray:
    # alexa (2007) Linear combination of transformations, appendix C
    # TODO: tests
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    k = 0
    I = np.eye(A.shape[0])
    while np.abs(A - I).max() > 0.5:
        A = sqrtm_db(A)
        k += 1
    A = I - A
    Z = A
    X = A
    i = 1
    while np.abs(Z).max() > EPS:
        Z = Z @ A
        i += 1
        X += Z / i
    return X * 2 ** k


def savetxt(fn, X, delims, header, footer):
    X = np.asanyarray(X)

    def write_mat(X, delims):
        for j, t in enumerate(X):
            if len(t.shape) == 0:
                fout.write(str(t))
            else:
                write_mat(t, delims[1:])
            if j != X.shape[0] - 1:
                fout.write(delims[0])

    assert len(X.shape) == len(delims)
    with open(fn, 'w') as fout:
        fout.write(header)
        write_mat(X, delims)
        fout.write(footer)
