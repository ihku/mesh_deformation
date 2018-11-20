#!/usr/bin/env python3

import numpy as np


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
