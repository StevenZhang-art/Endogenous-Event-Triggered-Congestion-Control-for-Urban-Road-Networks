import pandas as pd
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from multiprocessing import Pool, cpu_count
import os
import time

NETWORK_NAME = "Philadelphia"

NODE_CSV = f"{NETWORK_NAME}_node.csv"
NET_CSV = f"{NETWORK_NAME}_net.csv"
OD_CSV = f"{NETWORK_NAME}_od.csv"
OUTPUT_PKL = f"od_candidate_paths_{NETWORK_NAME.lower()}.pkl"
K_PATHS = 3
NUM_WORKERS = None
CHECKPOINT_PKL = f"{OUTPUT_PKL}.checkpoint"
CHECKPOINT_INTERVAL = 500
PENALTY_FACTOR = 1000.0

MILE_TO_METER = 1609.34
MINUTE_TO_SECOND = 60

_worker_row_idx = None
_worker_col_idx = None
_worker_base_weights = None
_worker_base_matrix = None
_worker_num_nodes = None
_worker_node_to_matrix_idx = None
_worker_node_list = None
_worker_edge_id_to_idx = None
_worker_node_out_edges = None
_worker_k_paths = None


def build_sparse_graph(net_df):
    edge_id_to_idx = {}
    node_out_edges = {}
    node_ids = set()

    sources = []
    targets = []
    weights = []

    for idx, row in enumerate(net_df.itertuples(index=False)):
        source = int(row.source)
        target = int(row.target)
        edge_id = f"{source}_{target}"
        edge_id_to_idx[edge_id] = idx
        node_ids.add(source)
        node_ids.add(target)

        length_m = row.length * MILE_TO_METER
        free_flow_time_s = row.freeFlowTime * MINUTE_TO_SECOND
        if free_flow_time_s > 0:
            speed = length_m / free_flow_time_s
        else:
            speed = length_m if length_m > 0 else 1.0

        weight = row.length / speed if speed > 0 else row.length

        sources.append(source)
        targets.append(target)
        weights.append(weight)
        node_out_edges.setdefault(source, {})[target] = edge_id

    node_list = sorted(node_ids)
    node_to_matrix_idx = {node_id: i for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)

    row_idx = np.array([node_to_matrix_idx[s] for s in sources], dtype=np.int64)
    col_idx = np.array([node_to_matrix_idx[t] for t in targets], dtype=np.int64)
    base_weights = np.array(weights, dtype=np.float64)

    return row_idx, col_idx, base_weights, num_nodes, node_to_matrix_idx, node_list, edge_id_to_idx, node_out_edges


def node_path_to_edge_indices(node_path, node_out_edges, edge_id_to_idx):
    edge_indices = []
    for i in range(len(node_path) - 1):
        s = node_path[i]
        d = node_path[i + 1]
        if s in node_out_edges and d in node_out_edges[s]:
            edge_id = node_out_edges[s][d]
            edge_indices.append(edge_id_to_idx[edge_id])
    return edge_indices


def _reconstruct_path(source_idx, target_idx, predecessors, node_list):
    path_idx = []
    node = target_idx
    while node != -9999 and node != source_idx:
        path_idx.append(node)
        node = predecessors[node]
    if node != source_idx:
        return None
    path_idx.append(source_idx)
    path_idx.reverse()
    return path_idx


def _penalize_path(local_weights, path_idx, node_list, node_out_edges, edge_id_to_idx):
    for i in range(len(path_idx) - 1):
        u_node = node_list[path_idx[i]]
        v_node = node_list[path_idx[i + 1]]
        if u_node in node_out_edges and v_node in node_out_edges[u_node]:
            edge_id = node_out_edges[u_node][v_node]
            pos = edge_id_to_idx[edge_id]
            local_weights[pos] *= PENALTY_FACTOR


def k_diverse_paths_shared(source_idx, target_idx, base_dist, base_predecessors, row_idx, col_idx,
                            base_weights, num_nodes, node_list, node_out_edges, edge_id_to_idx, k_paths):
    node_paths = []

    if not np.isfinite(base_dist[target_idx]):
        return node_paths

    path_idx = _reconstruct_path(source_idx, target_idx, base_predecessors, node_list)
    if path_idx is None:
        return node_paths

    node_paths.append([node_list[i] for i in path_idx])

    local_weights = base_weights.copy()
    _penalize_path(local_weights, path_idx, node_list, node_out_edges, edge_id_to_idx)

    for _ in range(k_paths - 1):
        if len(node_paths) >= k_paths:
            break

        matrix = csr_matrix((local_weights, (row_idx, col_idx)), shape=(num_nodes, num_nodes))
        dist, predecessors = dijkstra(matrix, directed=True, indices=source_idx, return_predecessors=True)

        if not np.isfinite(dist[target_idx]):
            break

        path_idx = _reconstruct_path(source_idx, target_idx, predecessors, node_list)
        if path_idx is None:
            break

        node_paths.append([node_list[i] for i in path_idx])

        if len(node_paths) >= k_paths:
            break

        _penalize_path(local_weights, path_idx, node_list, node_out_edges, edge_id_to_idx)

    while node_paths and len(node_paths) < k_paths:
        node_paths.append(node_paths[-1])

    return node_paths


def _init_worker(net_df, k_paths):
    global _worker_row_idx, _worker_col_idx, _worker_base_weights, _worker_base_matrix, _worker_num_nodes
    global _worker_node_to_matrix_idx, _worker_node_list, _worker_edge_id_to_idx
    global _worker_node_out_edges, _worker_k_paths
    (_worker_row_idx, _worker_col_idx, _worker_base_weights, _worker_num_nodes,
     _worker_node_to_matrix_idx, _worker_node_list, _worker_edge_id_to_idx,
     _worker_node_out_edges) = build_sparse_graph(net_df)
    _worker_base_matrix = csr_matrix((_worker_base_weights, (_worker_row_idx, _worker_col_idx)),
                                      shape=(_worker_num_nodes, _worker_num_nodes))
    _worker_k_paths = k_paths


def _solve_origin_group(origin_group):
    o_node, d_nodes = origin_group
    results = []

    if o_node not in _worker_node_to_matrix_idx:
        for d_node in d_nodes:
            results.append((o_node, d_node, None))
        return results

    source_idx = _worker_node_to_matrix_idx[o_node]
    base_dist, base_predecessors = dijkstra(_worker_base_matrix, directed=True, indices=source_idx,
                                             return_predecessors=True)

    for d_node in d_nodes:
        if d_node not in _worker_node_to_matrix_idx:
            results.append((o_node, d_node, None))
            continue

        target_idx = _worker_node_to_matrix_idx[d_node]
        node_paths = k_diverse_paths_shared(
            source_idx, target_idx, base_dist, base_predecessors, _worker_row_idx, _worker_col_idx,
            _worker_base_weights, _worker_num_nodes, _worker_node_list, _worker_node_out_edges,
            _worker_edge_id_to_idx, _worker_k_paths,
        )
        if not node_paths:
            results.append((o_node, d_node, None))
            continue

        path_idx_list = []
        for node_path in node_paths:
            edge_indices = node_path_to_edge_indices(node_path, _worker_node_out_edges, _worker_edge_id_to_idx)
            if len(edge_indices) > 0:
                path_idx_list.append(edge_indices)

        if not path_idx_list:
            results.append((o_node, d_node, None))
            continue

        while len(path_idx_list) < _worker_k_paths:
            path_idx_list.append(path_idx_list[-1])

        results.append((o_node, d_node, path_idx_list))

    return results

def _load_checkpoint():
    try:
        with open(CHECKPOINT_PKL, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def _save_checkpoint(od_candidate_paths):
    tmp_path = CHECKPOINT_PKL + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(od_candidate_paths, f)
    for attempt in range(5):
        try:
            os.replace(tmp_path, CHECKPOINT_PKL)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1.0)

def generate_candidate_paths(net_df, od_df, k_paths=K_PATHS, num_workers=None):
    _, _, _, _, _, _, edge_id_to_idx, _ = build_sparse_graph(net_df)
    num_edges = len(net_df)

    od_pairs = []
    seen_pairs = set()
    for row in od_df.itertuples(index=False):
        o_node = int(row.O)
        d_node = int(row.D)
        if o_node == d_node:
            continue
        pair = (o_node, d_node)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        od_pairs.append(pair)

    od_candidate_paths = _load_checkpoint()
    processed_count = 0
    skipped_count = 0

    pending_pairs = [pair for pair in od_pairs if pair not in od_candidate_paths]

    if not pending_pairs:
        return {
            "od_candidate_paths": od_candidate_paths,
            "edge_id_to_idx": edge_id_to_idx,
            "num_edges": num_edges,
        }, len(od_candidate_paths), skipped_count

    pending_by_origin = {}
    for o_node, d_node in pending_pairs:
        pending_by_origin.setdefault(o_node, []).append(d_node)
    origin_groups = list(pending_by_origin.items())

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    num_workers = min(num_workers, len(origin_groups))
    chunksize = max(1, len(origin_groups) // (num_workers * 4))

    since_last_checkpoint = 0
    with Pool(processes=num_workers, initializer=_init_worker, initargs=(net_df, k_paths)) as pool:
        for group_results in pool.imap_unordered(_solve_origin_group, origin_groups, chunksize=chunksize):
            for o_node, d_node, path_idx_list in group_results:
                if path_idx_list is None:
                    skipped_count += 1
                    continue
                od_candidate_paths[(o_node, d_node)] = path_idx_list
                processed_count += 1
                since_last_checkpoint += 1
                if since_last_checkpoint >= CHECKPOINT_INTERVAL:
                    _save_checkpoint(od_candidate_paths)
                    since_last_checkpoint = 0

    if since_last_checkpoint > 0:
        _save_checkpoint(od_candidate_paths)


    return {
        "od_candidate_paths": od_candidate_paths,
        "edge_id_to_idx": edge_id_to_idx,
        "num_edges": num_edges,
    }, processed_count, skipped_count


def main():
    net_df = pd.read_csv(NET_CSV)
    od_df = pd.read_csv(OD_CSV)

    print(f"Loaded {len(net_df)} links and {len(od_df)} requested OD pairs for {NETWORK_NAME}")
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else max(1, cpu_count() - 1)
    print(f"Computing up to {K_PATHS} candidate paths per OD pair via sparse Dijkstra with edge penalization ({num_workers} parallel workers)")

    result, processed_count, skipped_count = generate_candidate_paths(net_df, od_df, K_PATHS, num_workers)

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(result, f)
    if os.path.exists(CHECKPOINT_PKL):
       os.remove(CHECKPOINT_PKL)
    print(f"Processed {processed_count} OD pairs, skipped {skipped_count} unreachable OD pairs")
    print(f"Candidate paths saved to {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
