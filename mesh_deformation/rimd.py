#!/usr/bin/env python3

from collections import defaultdict

import tqdm
import numpy as np
import scipy.sparse
import scipy.linalg
from sksparse.cholmod import cholesky

from .mesh import Mesh, edges_to_adjacency_list
from .utils import savetxt

def cosv(a, b):
    return np.sum(a * b) / ((a ** 2).sum() * (b ** 2).sum()) ** 0.5


def _compute_coefs(mesh: Mesh) -> dict:
    vertices = mesh.get_vertices()
    res = defaultdict(lambda: 0)
    for poly in mesh.get_polygons():
        for i in range(3):
            u, v, w = poly[i], poly[i - 2], poly[i - 1]
            p1, p2, p3 = vertices[u], vertices[v], vertices[w]
            c = cosv(p2 - p3, p1 - p3)
            t = c / (1 - c ** 2) ** 0.5
            res[u, v] += t
            res[v, u] += t
    return res


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
    decomps = dict()
    cotans_dict = _compute_coefs(meshes[0])
    for u, vs in group_tuples(sorted(edges.keys())):
        centers = np.array([mesh.get_vertex(u) for mesh in meshes])
        adj = np.array([[mesh.get_vertex(v) for v in vs]
                        for mesh in meshes])
        cotans = np.array([cotans_dict[u, v] for v in vs], dtype=np.float64)
        # float64[n_meshes, n_adj, 3]
        diffs = centers[:, None, :] - adj
        diffs0 = diffs[0]
        # linear regression formula
        # BUG: swap diffs and diffs0
        tmp = np.linalg.inv(np.einsum('ai,aj->ij', diffs0 * cotans[:, None], diffs0))
        coefs = np.einsum('ij,aj,mak->mik', tmp, diffs0 * cotans[:, None], diffs)
        decomp = [scipy.linalg.polar(coefs[i]) for i in range(coefs.shape[0])]
        assert u not in decomps
        decomps[u] = decomp

    # TODO: understand why all rs are ids
    # TODO: more checks
    savetxt('rs1.txt', np.array([np.array(decomps[i])[:, 0] for i in range(meshes[0].get_num_vertices())])
            .transpose((1, 0, 2, 3)),
            (']]]\n\n\n[[[', ']],\n\n[[', '],\n[', ', '), '[[[', ']]]')

    savetxt('ss1.txt', np.array([np.array(decomps[i])[:, 1] for i in range(meshes[0].get_num_vertices())])
            .transpose((1, 0, 2, 3)),
            (']]]\n\n\n[[[', ']],\n\n[[', '],\n[', ', '), '[[[', ']]]')

    features = []
    n_vert = meshes[0].get_num_vertices()
    for m in tqdm.trange(len(meshes)):
        # float64[n_vert, 3, 3]
        rs = np.array([decomps[v][m][0] for v in range(n_vert)])
        ss = np.array([decomps[v][m][1] for v in range(n_vert)])

        feature = []
        # since s is symmetrical
        for s in ss:
            feature.extend([s[0, 0], s[0, 1], s[0, 2], s[1, 1], s[1, 2], s[2, 2]])

        for i, j in sorted(edges.keys()):
            if i < j:  # need only unique edges
                dr = rs[i].T @ rs[j]
                # assert np.allclose(dr, dr.T)
                logdr, err = scipy.linalg.logm(dr, disp=False)
                feature.extend([logdr[0, 0], logdr[0, 1], logdr[0, 2],
                                logdr[1, 1], logdr[1, 2], logdr[2, 2]])
        features.append(feature)

    return np.array(features, dtype=np.float64)


def rimds_to_meshes(rimd, mesh0: Mesh):
    rimd = np.asanyarray(rimd, np.float64)

    # float64[n_meshes, n_features]
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
                     for t in range(len(edges_k))]
                    for i in range(n_meshes)])
    cotans_dict = _compute_coefs(mesh0)

    # perform Cholesky factorization of matrix A
    A = scipy.sparse.lil_matrix((3 * n_vertices, 3 * n_vertices), dtype=np.float64)
    adj_list = edges_to_adjacency_list(edges.keys())
    for v, v_edges in adj_list.items():
        s = 0
        for u in v_edges:
            t = cotans_dict[v, u]
            A[v * 3, u * 3] = -t
            A[v * 3 + 1, u * 3 + 1] = -t
            A[v * 3 + 2, u * 3 + 2] = -t
            s += t
        A[v * 3, v * 3] = s
        A[v * 3 + 1, v * 3 + 1] = s
        A[v * 3 + 2, v * 3 + 2] = s

    # TODO: explore time of numpy vs scipy implementations
    # TODO: what to do if A isn't positive definite
    A_left = A[3:, :3]
    b_sub = A_left @ vert0[0]
    A = A[3:, 3:]
    print('cholesky started')
    # L = scipy.linalg.cholesky(A.toarray())
    A_factor = cholesky(A)
    print('cholesky finished')

    def optimize(ss, drs):
        # first, initialize parameters
        # float64[n_vertices, 3]
        vertices = np.empty_like(vert0)  # initialization isn't required since first step is a global step
        vertices[0] = vert0[0]
        rs = np.array([np.eye(3) for i in range(n_vertices)])  # TODO: advanced initialization with BFS

        drs_ = dict()
        for i, (u, v) in enumerate(edges_k):
            drs_[u, v] = drs[i]
            drs_[v, u] = drs[i].T
        drs = drs_

        def update_b():
            mat_ = np.zeros(shape=(n_vertices * 3, 3))
            for u, u_edges in adj_list.items():
                for v in u_edges:
                    mat_[u * 3:(u + 1) * 3] += rs[v] @ drs[v, u] @ ss[u]
            b = np.zeros(shape=(n_vertices * 3,))
            for u, u_edges in adj_list.items():
                for v in u_edges:
                    b[u * 3:(u + 1) * 3] += (mat_[v * 3:(v + 1) * 3] / len(adj_list[v])
                                             + mat_[u * 3:(u + 1) * 3] / len(adj_list[u])) \
                                            @ (vert0[u] - vert0[v]) * cotans_dict[u, v]
            return b[3:] / 2 - b_sub

        b = update_b()

        # perform optimization steps
        # TODO: explore convergence and choose number of steps properly
        N_ITERS = 15
        for i in tqdm.trange(N_ITERS):
            # global step
            # vertices[1:] = scipy.linalg.cho_solve((L, False), b).reshape(-1, 3)
            vertices[1:] = A_factor(b).reshape(-1, 3)
            # local step
            # TODO: weights
            mat_ = np.zeros(shape=(n_vertices * 3, 3))
            for u, u_edges in adj_list.items():
                for v in u_edges:
                    mat_[u * 3:(u + 1) * 3] += np.outer(vert0[u] - vert0[v], vertices[u] - vertices[v]) * cotans_dict[u, v]
            Q = np.zeros(shape=(n_vertices * 3, 3))
            for u, u_edges in adj_list.items():
                for v in u_edges:
                    Q[u * 3:(u + 1) * 3] += drs[u, v] @ ss[v] @ mat_[v * 3:(v + 1) * 3] / len(adj_list[v])
            for i in range(n_vertices):
                Q_i = Q[i * 3:(i + 1) * 3]
                u, s, vh = np.linalg.svd(Q_i)
                rs[i] = (u @ vh).T
                if np.linalg.det(rs[i]) < 0:
                    rs[i] = -rs[i]
                # TODO: refactor arrays from (n * 3, 3) to (n, 3, 3)
            b = update_b()

        savetxt('rs2.txt', rs,
                (']],\n\n[[', '],\n[', ', '), '[[[', ']]]')

        return Mesh(vertices, mesh0.get_polygons().copy())

    return [optimize(ss[i], drs[i]) for i in range(n_meshes)]


def write_rimd(path, rimd):
    np.save(path, rimd)

def read_rimd(path):
    return np.load(path)
