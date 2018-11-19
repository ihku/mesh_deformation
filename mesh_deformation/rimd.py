#!/usr/bin/env python3

import tqdm
import numpy as np
import scipy.sparse
import scipy.linalg

from .mesh import Mesh, edges_to_adjacency_list


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
            if i < j:  # need only unique edges
                dr = rs[i].T @ rs[j]
                logdr, err = scipy.linalg.logm(dr, disp=False)
                feature.extend([logdr[0, 0], logdr[0, 1], logdr[0, 2],
                                logdr[1, 1], logdr[1, 2], logdr[2, 2]])
        features.append(feature)

    return features


def rimd_to_meshes(rimd, mesh0: Mesh):
    # float32[n_meshes, n_features]
    assert len(rimd.shape) == 2

    def matr33sym(ar):
        assert ar.shape == (6,)
        return np.array([[ar[0], ar[1], ar[2]], [ar[1], ar[3], ar[4]], [ar[2], ar[4], ar[5]]])

    vert0 = mesh0.get_vertices()
    n_vertices = mesh0.get_num_vertices()
    n_meshes = rimd.shape[0]
    ss = np.array([[matr33sym(rimd[i, v * 6:(v + 1) * 6]) for v in range(n_vertices)] for i in range(n_meshes)])
    edges = mesh0.get_edges()
    edges_k = list(sorted((u, v) for u, v in edges.keys() if u < v))
    rimd_ = rimd[:, n_vertices * 6:]
    drs = np.array([[scipy.linalg.expm(matr33sym(rimd_[i, t * 6:(t + 1) * 6]))
                     for t in range(len(edges_k) // 2)]
                    for i in range(n_meshes)])

    def optimize(ss, drs):
        # TODO: implement function
        # first, initialize parameters
        # float32[n_vertices, 3]
        vertices = np.empty_like(vert0)  # initialization isn't required since first step is a global step
        rs = np.array([np.eye(3) for i in range(n_vertices)])  # TODO: advanced initialization with BFS

        drs_ = dict()
        for i, (u, v) in enumerate(edges_k):
            drs_[(u, v)] = drs[i]
            drs_[(v, u)] = drs[i].T
        drs = drs_

        # second, perform Cholesky factorization of matrix A
        # TODO: weights
        A = scipy.sparse.csc_matrix((3 * n_vertices, 3 * n_vertices), dtype=np.float32)
        adj_list = edges_to_adjacency_list(edges.keys())
        for v, v_edges in adj_list.items():
            A[v * 3, v * 3] = len(v_edges)
            A[v * 3 + 1, v * 3 + 1] = len(v_edges)
            A[v * 3 + 2, v * 3 + 2] = len(v_edges)
            for u in v_edges:
                A[v * 3, u * 3] = len(v_edges)
                A[v * 3 + 1, u * 3 + 1] = len(v_edges)
                A[v * 3 + 2, u * 3 + 2] = len(v_edges)

        # TODO: explore time of numpy vs scipy implementations
        L = scipy.linalg.cholesky(A.toarray())

        mat_ = np.zeros(shape=(n_vertices * 3, 3))
        for u, u_edges in adj_list.items():
            for v in u_edges:
                mat_[u * 3:(u + 1) * 3] += rs[v] @ drs[(v, u)] @ ss[u]
        b = np.zeros(shape=(n_vertices * 3,))
        for u, u_edges in adj_list.items():
            for v in u_edges:
                b[u * 3:(u + 1) * 3] += (mat_[v * 3: (v + 1) * 3] / len(adj_list[v])
                                         + mat_[u * 3: (u + 1) * 3] / len(adj_list[u])) @ (vert0[u] - vert0[v])

        # third, perform optimization steps
        # TODO: explore convergence and choose number of steps properly
        N_ITERS = 10
        for i in range(N_ITERS):
            # global step
            vertices[:] = scipy.linalg.cho_solve((L, False), b)
            # local step
            mat_ = np.zeros(shape=(n_vertices * 3, 3))
            for u, u_edges in adj_list.items():
                for v in u_edges:
                    mat_[u * 3:(u + 1) * 3] += np.outer(vert0[u] - vert0[v], vertices[u] - vertices[v])
            Q = np.zeros(shape=(n_vertices * 3, 3))
            for u, u_edges in adj_list.items():
                for v in u_edges:
                    Q[u * 3:(u + 1) * 3] += drs[u, v] @ ss[v] @ mat_[v * 3:(v + 1) * 3] / len(adj_list[v])
            for i in range(n_vertices):
                Q_i = Q[i * 3:(i + 1) * 3]
                u, s, vh = np.linalg.svd(Q_i)
                rs[i] = (u @ vh).T
                # TODO: refactor arrays from (n * 3, 3) to (n, 3, 3)

        return Mesh(vertices, mesh0.get_polygons().copy())

    return [optimize(ss[i], drs[i]) for i in range(n_meshes)]
