#!/usr/bin/env python3

import tqdm
import numpy as np
import scipy.linalg

from .mesh import Mesh

def meshes_to_rimds(meshes):
    # need to check if meshes are similar
    
    def group_tuples(seq):
        v = None
        adj = []
        for edge in seq:
            if v is None:
                v = edge[0]
                adj.append(edge[1])
            elif v == edge[0]:
                adj.append(edge[1])
            else:
                yield v, adj
                v = edge[0]
                adj = [edge[1]]
        if len(adj) > 0:
            yield v, adj
    
    edges = meshes[0].get_edges()
    features = [[] for i in range(len(meshes))]
    decomps = dict()
    for u, vs in group_tuples(sorted(edges.keys())):
        faces = np.array([edges[(u, v)] for v in vs])
        centers = np.array([mesh.get_vertex(u) for mesh in meshes])
        adj = np.array([[mesh.get_vertex(v) for v in vs]
                            for mesh in meshes])
        # float32[n_meshes, n_adj, 3]
        diffs = centers[:, None, :] - adj
        diffs0 = diffs[0]
        # TODO: weights
        # linear regression formula
        tmp = np.linalg.inv(np.einsum('mai,maj->mij', diffs, diffs))
        coefs = np.einsum('mij,maj,ak->mik', tmp, diffs, diffs0)
        decomp = [scipy.linalg.polar(coefs[i]) for i in range(coefs.shape[0])]
        assert u not in decomps
        decomps[u] = decomp
    
    features = []
    n_vert = meshes[0].get_num_vertices()
    for m in tqdm.trange(len(meshes)):
        # float32[n_vert, 3, 3]
        rs = np.array([decomps[v][m][0] for v in range(n_vert)])
        ss = np.array([decomps[v][m][1] for v in range(n_vert)])
        
        feature = []
        # since s is symmetrical
        for s in ss:
            feature.extend([s[0, 0], s[0, 1], s[0, 2], s[1, 1], s[1, 2], s[2, 2]])
        
        for i, j in sorted(edges.keys()):
            if i < j: # need only unique edges
                dr = rs[i].T @ rs[j]
                logdr, err = scipy.linalg.logm(dr, disp=False)
                feature.extend([logdr[0, 0], logdr[0, 1], logdr[0, 2],
                                logdr[1, 1], logdr[1, 2], logdr[2, 2]])
        features.append(feature)
    
    return features

def rimd_to_meshes(rimd, mesh0):
    # float32[n_meshes, n_features]
    assert len(rimd.shape) == 2
    
    def matr33sym(ar):
        assert ar.shape == (6,)
        return np.array([[ar[0], ar[1], ar[2]], [ar[1], ar[3], ar[4]], [ar[2], ar[4], ar[5]]])
    
    n_vertices = mesh0.get_num_vertices()
    n_meshes = rimd.shape[0]
    ss = np.array([[matr33sym(rimd[i, v*6:(v+1)*6]) for v in range(n_vertices)] for i in range(n_meshes)])
    edges = mesh0.get_edges()
    rimd_ = rimd[:, n_vertices*6:]
    drs = np.array([[scipy.linalg.expm(matr33sym(rimd_[i, t*6:(t+1)*6]))
                        for t in range(len(edges.keys()) // 2)]
                        for i in range(n_meshes)])
    
    def optimize(ss, drs):
        # TODO: implement function
        return Mesh([], [])
    
    return [optimize(ss[i], drs[i]) for i in range(n_meshes)]
