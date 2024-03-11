from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np
import random

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]
_OutputClass = specs.OutputClass


def sample_indices_with_weights(A):
    if np.sum(A) == 0:
        breakpoint()
        return -1, -1
    A_normalized = A / np.sum(A)

    flat_indices = np.arange(A.size).reshape(-1)
    probabilities = A_normalized.reshape(-1)
    sampled_index = np.random.choice(flat_indices, p=probabilities)

    i, j = np.unravel_index(sampled_index, A.shape)

    return i, j


def sample_lexicographically_smallest(A):
    non_zero_indices = np.transpose(np.nonzero(A))
    if len(non_zero_indices) == 0:
        return -1, -1
    sorted_indices = sorted(non_zero_indices, key=lambda x: (x[0], x[1]))
    return sorted_indices[0]


def replace_edges(adj_matrix, i, j):
    adj_matrix[i, :] += adj_matrix[j, :]
    adj_matrix[:, i] += adj_matrix[:, j]

    adj_matrix[j, :] = 0
    adj_matrix[:, j] = 0

    adj_matrix[i, j] = 0
    adj_matrix[j, i] = 0
    adj_matrix[i, i] = 0
    adj_matrix[j, j] = 0

    return adj_matrix


def karger(A: _Array, Seed: int) -> _Out:
    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['karger'])
    A_pos = np.arange(A.shape[0])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'seed': Seed
        })
    random.seed(Seed)
    np.random.seed(Seed)
    group = np.arange(A.shape[0])
    graph_comp = np.copy(A)
    for s in range(A.shape[0] - 2):
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'group_h': np.copy(group),
                'graph_comp': np.copy(graph_comp),
            })

        i, j = sample_indices_with_weights(graph_comp)
        assert (i != j)
        assert (i != -1)

        i = group[i]
        j = group[j]
        if A_pos[i] > A_pos[j]:
            tmp = i
            i = j
            j = tmp
        group[group == j] = i
        replace_edges(graph_comp, i, j)

    probing.push(probes, specs.Stage.OUTPUT, next_probe={'group': np.copy(group)})
    probing.finalize(probes)
    return group, probes


def karger_deterministic(A: _Array) -> _Out:
    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['karger_deterministic'])
    A_pos = np.arange(A.shape[0])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
        })
    group = np.arange(A.shape[0])
    graph_comp = np.copy(A)
    for s in range(A.shape[0] - 2):
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'group_h': np.copy(group),
                'graph_comp': np.copy(graph_comp),
            })

        i, j = sample_lexicographically_smallest(graph_comp)
        assert (i != j)
        assert (i != -1)

        i = group[i]
        j = group[j]
        if A_pos[i] > A_pos[j]:
            tmp = i
            i = j
            j = tmp
        group[group == j] = i
        replace_edges(graph_comp, i, j)

    probing.push(probes, specs.Stage.OUTPUT, next_probe={'group': np.copy(group)})
    probing.finalize(probes)
    return group, probes


def karger_kruskal(A: _Array, Seed: int) -> _Out:
    """Karger's algorithm via Kruskal's MST with random permutation."""

    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['karger_kruskal'])

    A_pos = np.arange(A.shape[0])
    random.seed(Seed)
    np.random.seed(Seed)

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A))
        })

    pi = np.arange(A.shape[0])

    def mst_union(u, v, in_mst, probes):
        root_u = u
        root_v = v

        mask_u = np.zeros(in_mst.shape[0])
        mask_v = np.zeros(in_mst.shape[0])

        mask_u[u] = 1
        mask_v[v] = 1

        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'in_mst_h': np.copy(in_mst),
                'pi': np.copy(pi),
                'u': probing.mask_one(u, A.shape[0]),
                'v': probing.mask_one(v, A.shape[0]),
                'root_u': probing.mask_one(root_u, A.shape[0]),
                'root_v': probing.mask_one(root_v, A.shape[0]),
                'mask_u': np.copy(mask_u),
                'mask_v': np.copy(mask_v),
                'phase': probing.mask_one(1, 3)
            })

        while pi[root_u] != root_u:
            root_u = pi[root_u]
            for i in range(mask_u.shape[0]):
                if mask_u[i] == 1:
                    pi[i] = root_u
            mask_u[root_u] = 1
            probing.push(
                probes,
                specs.Stage.HINT,
                next_probe={
                    'in_mst_h': np.copy(in_mst),
                    'pi': np.copy(pi),
                    'u': probing.mask_one(u, A.shape[0]),
                    'v': probing.mask_one(v, A.shape[0]),
                    'root_u': probing.mask_one(root_u, A.shape[0]),
                    'root_v': probing.mask_one(root_v, A.shape[0]),
                    'mask_u': np.copy(mask_u),
                    'mask_v': np.copy(mask_v),
                    'phase': probing.mask_one(1, 3)
                })

        while pi[root_v] != root_v:
            root_v = pi[root_v]
            for i in range(mask_v.shape[0]):
                if mask_v[i] == 1:
                    pi[i] = root_v
            mask_v[root_v] = 1
            probing.push(
                probes,
                specs.Stage.HINT,
                next_probe={
                    'in_mst_h': np.copy(in_mst),
                    'pi': np.copy(pi),
                    'u': probing.mask_one(u, A.shape[0]),
                    'v': probing.mask_one(v, A.shape[0]),
                    'root_u': probing.mask_one(root_u, A.shape[0]),
                    'root_v': probing.mask_one(root_v, A.shape[0]),
                    'mask_u': np.copy(mask_u),
                    'mask_v': np.copy(mask_v),
                    'phase': probing.mask_one(2, 3)
                })

        if root_u < root_v:
            in_mst[u, v] = 1
            in_mst[v, u] = 1
            pi[root_u] = root_v
        elif root_u > root_v:
            in_mst[u, v] = 1
            in_mst[v, u] = 1
            pi[root_v] = root_u
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'in_mst_h': np.copy(in_mst),
                'pi': np.copy(pi),
                'u': probing.mask_one(u, A.shape[0]),
                'v': probing.mask_one(v, A.shape[0]),
                'root_u': probing.mask_one(root_u, A.shape[0]),
                'root_v': probing.mask_one(root_v, A.shape[0]),
                'mask_u': np.copy(mask_u),
                'mask_v': np.copy(mask_v),
                'phase': probing.mask_one(0, 3)
            })

    in_mst = np.zeros((A.shape[0], A.shape[0]))

    # Prep to sort edge array
    lx = []
    ly = []
    wts = []
    num_edges = 0
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if A[i, j] > 0:
                num_edges += 1
    permutation = list(range(num_edges))
    random.shuffle(permutation)
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if A[i, j] > 0:
                lx.append(i)
                ly.append(j)
                wts.append(permutation[-1])
                permutation.pop()

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'in_mst_h': np.copy(in_mst),
            'pi': np.copy(pi),
            'u': probing.mask_one(0, A.shape[0]),
            'v': probing.mask_one(0, A.shape[0]),
            'root_u': probing.mask_one(0, A.shape[0]),
            'root_v': probing.mask_one(0, A.shape[0]),
            'mask_u': np.zeros(A.shape[0]),
            'mask_v': np.zeros(A.shape[0]),
            'phase': probing.mask_one(0, 3)
        })
    for ind in np.argsort(wts):
        u = lx[ind]
        v = ly[ind]
        mst_union(u, v, in_mst, probes)

    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={'in_mst': np.copy(in_mst)},
    )
    probing.finalize(probes)

    return in_mst, probes
