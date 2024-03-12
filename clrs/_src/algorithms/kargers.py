from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np
import random

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]
_OutputClass = specs.OutputClass


def tent_map(x):
    if 0 <= x <= 0.5:
        return 2 * x
    elif 0.5 < x <= 1:
        return 2 - 2 * x


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


def sample_indices_with_weights_array(A, w):
    n = A.shape[0]

    edge_weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            edge_weights[i, j] = (w[i]+0.05) * (w[j]+0.05) * A[i, j]

    total_prob = np.sum(edge_weights)
    if total_prob == 0:
        return -1, -1
    edge_weights /= total_prob

    i, j = np.unravel_index(np.argmax(edge_weights), edge_weights.shape)

    return i, j



def sample_lexicographically_smallest(A):
    non_zero_indices = np.transpose(np.nonzero(A))
    if len(non_zero_indices) == 0:
        return -1, -1
    sorted_indices = sorted(non_zero_indices, key=lambda x: (x[0], x[1]))
    return sorted_indices[0]


def sample_indices_of_largest_element(A):
    max_index = np.argmax(A)
    i, j = np.unravel_index(max_index, A.shape)
    return i, j


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


def replace_edges_max(adj_matrix, i, j):
    adj_matrix[i, :] = np.maximum(adj_matrix[i, :], adj_matrix[j, :])
    adj_matrix[:, i] = np.maximum(adj_matrix[:, i], adj_matrix[:, j])

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
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'group_h': np.copy(group),
            'graph_comp': np.copy(graph_comp),
        })
    probing.push(probes, specs.Stage.OUTPUT, next_probe={'group': np.copy(group)})
    probing.finalize(probes)
    return group, probes


def karger_with_node_weights(A: _Array, w: _Array) -> _Out:
    chex.assert_rank(A, 2)
    chex.assert_rank(w, 1)
    probes = probing.initialize(specs.SPECS['karger_with_node_weights'])
    A_pos = np.arange(A.shape[0])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'node_weights': np.copy(w)
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
                'node_weights_h': np.copy(w)
            })

        i, j = sample_indices_with_weights_array(graph_comp, w)
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
        for i in range(A.shape[0]):
            w[i] = tent_map(w[i])

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'group_h': np.copy(group),
            'graph_comp': np.copy(graph_comp),
            'node_weights_h': np.copy(w)
        })
    probing.push(probes, specs.Stage.OUTPUT, next_probe={'group': np.copy(group)})
    probing.finalize(probes)
    return group, probes


def karger_kruskal(A: _Array, Seed: int) -> _Out:
    """Karger's algorithm via Kruskal's MST with random permutation."""

    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['karger_kruskal'])
    components = A.shape[0] - 2

    A_pos = np.arange(A.shape[0])
    random.seed(Seed)
    np.random.seed(Seed)

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'components': probing.mask_one(components, A.shape[0]),
            'zero': probing.mask_one(0, A.shape[0])
        })

    pi = np.arange(A.shape[0])
    pi_prev = np.copy(pi)

    def mst_union(u, v, n, probes, components):
        root_u = u
        root_v = v

        mask_u = np.zeros(n)
        mask_v = np.zeros(n)

        mask_u[u] = 1
        mask_v[v] = 1

        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'pi': np.copy(pi),
                'pi': np.copy(pi_prev),
                'u': probing.mask_one(u, A.shape[0]),
                'v': probing.mask_one(v, A.shape[0]),
                'root_u': probing.mask_one(root_u, A.shape[0]),
                'root_v': probing.mask_one(root_v, A.shape[0]),
                'mask_u': np.copy(mask_u),
                'mask_v': np.copy(mask_v),
                'phase': probing.mask_one(1, 3),
                'components_h': probing.mask_one(components, A.shape[0]),
                'zero_h': probing.mask_one(0, A.shape[0])
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
                    'pi': np.copy(pi),
                    'pi_prev': np.copy(pi_prev),
                    'u': probing.mask_one(u, A.shape[0]),
                    'v': probing.mask_one(v, A.shape[0]),
                    'root_u': probing.mask_one(root_u, A.shape[0]),
                    'root_v': probing.mask_one(root_v, A.shape[0]),
                    'mask_u': np.copy(mask_u),
                    'mask_v': np.copy(mask_v),
                    'phase': probing.mask_one(1, 3),
                    'components_h': probing.mask_one(components, A.shape[0]),
                    'zero_h': probing.mask_one(0, A.shape[0])
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
                    'pi': np.copy(pi),
                    'u': probing.mask_one(u, A.shape[0]),
                    'v': probing.mask_one(v, A.shape[0]),
                    'root_u': probing.mask_one(root_u, A.shape[0]),
                    'root_v': probing.mask_one(root_v, A.shape[0]),
                    'mask_u': np.copy(mask_u),
                    'mask_v': np.copy(mask_v),
                    'phase': probing.mask_one(2, 3),
                    'components_h': probing.mask_one(components, A.shape[0]),
                    'zero_h': probing.mask_one(0, A.shape[0])
                })
        success_ret = True
        if root_u > root_v and components > 0:
            pi[root_u] = root_v
            components -= 1
        elif root_u < root_v and components > 0:
            pi[root_v] = root_u
            components -= 1
        else:
            success_ret = False

        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'pi': np.copy(pi),
                'u': probing.mask_one(u, A.shape[0]),
                'v': probing.mask_one(v, A.shape[0]),
                'root_u': probing.mask_one(root_u, A.shape[0]),
                'root_v': probing.mask_one(root_v, A.shape[0]),
                'mask_u': np.copy(mask_u),
                'mask_v': np.copy(mask_v),
                'phase': probing.mask_one(0, 3),
                'components_h': probing.mask_one(components, A.shape[0]),
                'zero_h': probing.mask_one(0, A.shape[0])
            })
        return success_ret

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
            'pi': np.copy(pi),
            'u': probing.mask_one(0, A.shape[0]),
            'v': probing.mask_one(0, A.shape[0]),
            'root_u': probing.mask_one(0, A.shape[0]),
            'root_v': probing.mask_one(0, A.shape[0]),
            'mask_u': np.zeros(A.shape[0]),
            'mask_v': np.zeros(A.shape[0]),
            'phase': probing.mask_one(0, 3),
            'components_h': probing.mask_one(components, A.shape[0]),
            'zero_h': probing.mask_one(0, A.shape[0])
        })
    for ind in np.argsort(wts):
        u = lx[ind]
        v = ly[ind]
        success = mst_union(u, v, A.shape[0], probes, components)
        if success: components -= 1

    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            'pi_final': np.copy(pi),
        },
    )
    probing.finalize(probes)
    return pi, probes


def karger_kruskal_naive(A: _Array, random_weights: _Array) -> _Out:
    chex.assert_rank(A, 2)
    chex.assert_rank(random_weights, 2)
    probes = probing.initialize(specs.SPECS['karger_kruskal_naive'])
    A_pos = np.arange(A.shape[0])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'random_weights': np.copy(random_weights),
        })
    n = A.shape[0]
    group = np.arange(n)
    graph_comp = np.copy(np.multiply(A, random_weights))
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'group_h': np.copy(group),
            'graph_comp': np.copy(graph_comp),
            'u': probing.mask_one(0, n),
            'v': probing.mask_one(0, n),
            'phase': probing.mask_one(0, 3),
        })
    for s in range(A.shape[0] - 2):

        i, j = sample_indices_of_largest_element(graph_comp)
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'group_h': np.copy(group),
                'graph_comp': np.copy(graph_comp),
                'u': probing.mask_one(i, n),
                'v': probing.mask_one(j, n),
                'phase': probing.mask_one(1, 3),
            })
        assert (i != j)
        assert (i != -1)

        i = group[i]
        j = group[j]
        if A_pos[i] > A_pos[j]:
            tmp = i
            i = j
            j = tmp
        group[group == j] = i
        replace_edges_max(graph_comp, i, j)
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'group_h': np.copy(group),
                'graph_comp': np.copy(graph_comp),
                'u': probing.mask_one(i, n),
                'v': probing.mask_one(j, n),
                'phase': probing.mask_one(2, 3),
            })
    probing.push(probes, specs.Stage.OUTPUT, next_probe={'group': np.copy(group)})
    probing.finalize(probes)
    return group, probes


def karger_prim(A: _Array, random_weights: _Array, s: int = 0) -> _Out:
    """Karger's implementation based on Prim's minimum spanning tree algorithm (Prim, 1957)."""

    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['karger_prim'])

    A_pos = np.arange(A.shape[0])

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'random_weights': np.copy(random_weights)
        })
    A = np.multiply(A, random_weights)
    key = np.zeros(A.shape[0])
    mark = np.zeros(A.shape[0])
    in_queue = np.zeros(A.shape[0])
    group = np.arange(A.shape[0])
    key[s] = 0
    group_rep = 0
    in_queue[s] = 1
    max_edge = 0
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'group_h': np.copy(group),
            'key': np.copy(key),
            'mark': np.copy(mark),
            'in_queue': np.copy(in_queue),
            'u': probing.mask_one(s, A.shape[0]),
            'phase': probing.mask_one(0, 3),
            'max_edge': max_edge,
            'group_rep': probing.mask_one(group_rep, A.shape[0])
        })

    for _ in range(A.shape[0]):
        u = np.argsort(key + (1.0 - in_queue) * 1e9)[0]  # drop-in for extract-min
        phase = 1
        if key[u] > max_edge:
            phase = 2
            max_edge = key[u]
        if in_queue[u] == 0:
            break
        mark[u] = 1
        in_queue[u] = 0
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'group_h': np.copy(group),
                'key': np.copy(key),
                'mark': np.copy(mark),
                'in_queue': np.copy(in_queue),
                'u': probing.mask_one(u, A.shape[0]),
                'phase': probing.mask_one(phase, 3),
                'max_edge': max_edge,
                'group_rep': probing.mask_one(group_rep, A.shape[0])
            })
        if phase == 2:
            group[u] = u
        for v in range(A.shape[0]):
            if A[u, v] >= 1e-9:
                if mark[v] == 0 and (in_queue[v] == 0 or A[u, v] < key[v]):
                    group[v] = group[u]
                    key[v] = A[u, v]
                    in_queue[v] = 1
            if group[v] == group_rep and phase == 2:
                group[v] = group[s]
        if phase == 2:
            group_rep = u
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'group_h': np.copy(group),
                'key': np.copy(key),
                'mark': np.copy(mark),
                'in_queue': np.copy(in_queue),
                'u': probing.mask_one(u, A.shape[0]),
                'phase': probing.mask_one(0, 3),
                'max_edge': max_edge,
                'group_rep': probing.mask_one(group_rep, A.shape[0])
            })

    probing.push(probes, specs.Stage.OUTPUT, next_probe={'group': np.copy(group)})
    probing.finalize(probes)

    return group, probes


if __name__ == '__main__':
    adj_matrix = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    weights = np.random.rand(adj_matrix.shape[0])

    print(adj_matrix, weights)

    print(karger_with_node_weights(adj_matrix, weights))
