


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from numba import jit
import gc
from collections import defaultdict, deque
from tqdm import tqdm
try:
    import libsumo as traci
    import libsumo.constants as tc
except ImportError:
    import traci
    import traci.constants as tc
import sumolib
import pickle
import pandas
import re
import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.sparse as sp
import matplotlib.font_manager as fm

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['axes.unicode_minus'] = False

try:
    fm.findfont("Times New Roman")
except:
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False

TRAFFIC_LOADING_MODEL = "SUMO_MICROSCOPIC_CARFOLLOWING"
WARMUP_DURATION = 300
CLEARANCE_DURATION = 600
LEAD_TIME_MAX_WINDOW = 1800
RISE_RESPONSE_MAX_WINDOW = 300
CALIBRATION_SEEDS = [100, 101, 102]
VALIDATION_SEEDS = [200, 201, 202]
TEST_SEEDS = [42]


class EndogenousCongestionTracker:
    def __init__(self,
                 num_edges,
                 congestion_threshold=0.4,
                 max_single_edge_ratio=0.2,
                 min_base_allocation=0.001,
                 time_window_size=60,
                 decay_factor=0.95,
                 gamma_max=1.0):
        self.num_edges = num_edges
        if np.isscalar(congestion_threshold):
            self.congestion_threshold = np.full(num_edges, congestion_threshold, dtype=np.float32)
        else:
            self.congestion_threshold = np.asarray(congestion_threshold, dtype=np.float32)
        self.max_single_edge_ratio = max_single_edge_ratio
        self.min_base_allocation = min_base_allocation
        self.time_window_size = time_window_size
        self.decay_factor = decay_factor
        self.gamma_max = gamma_max
        self.max_od_congested_ratio = gamma_max

        self.allocation_windows = np.zeros((num_edges, time_window_size), dtype=np.float32)
        self.current_time_idx = 0
        self.total_allocated_vehicles = 0
        self.edge_capacity = np.ones(num_edges, dtype=np.float32)

        self._decay_pows = self.decay_factor ** np.arange(time_window_size, dtype=np.float32)
        self._flat_total = num_edges * time_window_size

    def set_edge_capacity(self, capacity_array):
        self.edge_capacity = capacity_array.astype(np.float32)

    def step_forward(self):
        self.current_time_idx = (self.current_time_idx + 1) % self.time_window_size
        self.allocation_windows[:, self.current_time_idx] = 0.0

    def update_allocation(self, edge_indices, arrival_offsets, vehicle_weights=None):
        n = len(edge_indices)
        if n == 0:
            return

        weights = np.ones(n, dtype=np.float32) if vehicle_weights is None else vehicle_weights.astype(np.float32)
        offsets = np.clip(arrival_offsets, 0, self.time_window_size - 1).astype(np.int32)
        target_indices = (self.current_time_idx + offsets) % self.time_window_size

        flat_idx = edge_indices * self.time_window_size + target_indices
        aggregated = np.bincount(flat_idx, weights=weights, minlength=self._flat_total)
        self.allocation_windows += aggregated.reshape(self.num_edges, self.time_window_size)

        self.total_allocated_vehicles += np.sum(weights)

    def record_measured_arrival(self, edge_indices, measured_counts):
        edge_indices = np.asarray(edge_indices)
        if edge_indices.size == 0:
            return
        measured_counts = np.asarray(measured_counts, dtype=np.float32)
        valid = ~np.isnan(measured_counts)
        if not np.any(valid):
            return
        self.allocation_windows[edge_indices[valid], self.current_time_idx] = measured_counts[valid]

    def calculate_congestion_risk(self):
        decay_sum = np.sum(self._decay_pows)
        window_total = np.sum(self.allocation_windows * self._decay_pows[None, :], axis=1) / decay_sum
        congestion_risk = np.divide(window_total, self.edge_capacity,
                                    out=np.zeros_like(window_total),
                                    where=self.edge_capacity > 1e-6)
        congestion_risk = np.maximum(congestion_risk, self.min_base_allocation)
        return congestion_risk

    def get_congested_edges(self, congestion_risk=None):
        if congestion_risk is None:
            congestion_risk = self.calculate_congestion_risk()
        return set(np.where(congestion_risk >= self.congestion_threshold)[0])

    def get_congested_mask(self, congestion_risk=None):
        if congestion_risk is None:
            congestion_risk = self.calculate_congestion_risk()
        return congestion_risk >= self.congestion_threshold

    def get_od_congested_limit(self, total_demand):
        return total_demand * self.max_od_congested_ratio

    def get_single_edge_limit(self, total_demand):
        return total_demand * self.max_single_edge_ratio


@jit(nopython=True, cache=True)
def _numba_allocate_paths_core(n, demand, max_alloc, weights):
    allocation = np.zeros(n, dtype=np.float32)
    demand_f32 = np.float32(demand)
    max_alloc_f32 = np.float32(max_alloc)
    total_weight = np.float32(np.sum(weights))
    if total_weight < 1e-6:
        total_weight = np.float32(n)

    for i in range(n):
        base = min(demand_f32 * weights[i] / total_weight, max_alloc_f32)
        allocation[i] = base

    remain = demand_f32
    if remain > 1e-6:
        rw = (max_alloc_f32 - allocation) * weights
        rt = np.sum(rw)
        if rt < 1e-6:
            rt = 1.0
        for i in range(n):
            av = max_alloc_f32 - allocation[i]
            add = min(remain * rw[i] / rt, av)
            allocation[i] += add
    return allocation


class EndogenousRouteAllocator:
    def __init__(self, num_edges, od_candidate_paths,
                 congestion_threshold=0.4,
                 max_allocation_ratio=0.25,
                 smooth_window=60,
                 max_single_edge_ratio=0.2,
                 min_base_allocation=0.001,
                 time_decay_factor=0.95,
                 max_od_congested_ratio=None,
                 gamma_max=0.7,
                 congestion_update_interval=30,
                 use_d3stn_enhance=False,
                 feedback_gain=0.02):
        self.num_edges = num_edges
        self.od_candidate_paths = od_candidate_paths
        self.congestion_threshold = congestion_threshold
        self.max_allocation_ratio = max_allocation_ratio
        self.smooth_window = smooth_window

        self.allocate_count = np.zeros(num_edges, dtype=np.int32)
        self.cum_usage = np.zeros(num_edges, dtype=np.float32)
        self.total_count = 0
        self.total_demand = 0.0
        self.unmet_demand_total = 0.0

        self.edge_status = np.ones(num_edges, dtype=np.bool_)
        self.history_congestion_ratio = np.zeros((num_edges, smooth_window), dtype=np.float32)
        self.window_idx = 0

        self.feedback_gain = feedback_gain
        self.gamma_max = gamma_max if max_od_congested_ratio is None else max_od_congested_ratio

        if use_d3stn_enhance:
            self.congestion_tracker = D3STN_EnhancedTracker(
                num_edges=num_edges,
                congestion_threshold=congestion_threshold,
                max_single_edge_ratio=max_single_edge_ratio,
                min_base_allocation=min_base_allocation,
                time_window_size=smooth_window,
                decay_factor=time_decay_factor,
                gamma_max=self.gamma_max
            )
        else:
            self.congestion_tracker = EndogenousCongestionTracker(
                num_edges=num_edges,
                congestion_threshold=congestion_threshold,
                max_single_edge_ratio=max_single_edge_ratio,
                min_base_allocation=min_base_allocation,
                time_window_size=smooth_window,
                decay_factor=time_decay_factor,
                gamma_max=self.gamma_max
            )

        self._path_priority_cache = {}
        self._path_edge_ids_cache = {}
        self._path_edge_pos_cache = {}
        self._path_edges_cache = {}
        self._path_offsets_cache = {}
        self._path_congest_cache = {}
        self._path_reduce_idx_cache = {}

        self._congestion_update_interval = congestion_update_interval
        self._step_counter = 0
        self._cached_risk = None
        self._cached_mask = None

    def set_edge_capacity(self, capacity_array):
        self.congestion_tracker.set_edge_capacity(capacity_array)

    def init_d3stn_gnn(self, adjacency_matrix, edge_lengths, free_flow_speeds):
        if hasattr(self.congestion_tracker, 'build_hybrid_adjacency'):
            self.congestion_tracker.build_hybrid_adjacency(adjacency_matrix, edge_lengths, free_flow_speeds)

    def precompute_path_data(self, free_flow_speeds, edge_lengths, edge_idx_to_id):
        for od_key, paths in self.od_candidate_paths.items():
            num_paths = len(paths)
            weights = np.zeros(num_paths, dtype=np.float32)
            edge_ids_list = []
            edge_pos_list = []
            path_edges_list = []
            path_offsets_list = []
            all_edges_flat = []
            path_lengths = []

            for i, path in enumerate(paths):
                path_arr = np.array(path, dtype=np.int32)
                path_edges_list.append(path_arr)
                path_lengths.append(len(path_arr))
                all_edges_flat.extend(path_arr)

                free_flow_time = np.sum(edge_lengths[path_arr] / np.maximum(free_flow_speeds[path_arr], 1e-3))
                weights[i] = 1.0 / max(free_flow_time, 1e-3)

                offsets = np.cumsum(edge_lengths[path_arr] / np.maximum(free_flow_speeds[path_arr], 1e-3))
                path_offsets_list.append(offsets.astype(np.float32))

                edge_ids = [edge_idx_to_id[e] for e in path]
                edge_ids_list.append(edge_ids)
                edge_pos_list.append({eid: idx for idx, eid in enumerate(edge_ids)})

            self._path_priority_cache[od_key] = weights
            self._path_edge_ids_cache[od_key] = edge_ids_list
            self._path_edge_pos_cache[od_key] = edge_pos_list
            self._path_edges_cache[od_key] = path_edges_list
            self._path_offsets_cache[od_key] = path_offsets_list
            self._path_congest_cache[od_key] = (
                np.array(all_edges_flat, dtype=np.int32),
                np.array(path_lengths, dtype=np.int32)
            )
            if num_paths > 0:
                lens_arr = np.array(path_lengths, dtype=np.int32)
                reduce_idx = np.concatenate([[0], np.cumsum(lens_arr)[:-1]])
                self._path_reduce_idx_cache[od_key] = reduce_idx

    def update_edge_status(self, edge_indices, status):
        mask = (edge_indices >= 0) & (edge_indices < self.num_edges)
        valid = edge_indices[mask]
        self.edge_status[valid] = status
        self.allocate_count[valid[~status]] = 0
        self.cum_usage[~status] = 0.0
        self.congestion_tracker.allocation_windows[valid[~status]] = 0.0

    def allocate_vehicles(self, vehicle_list, free_flow_speeds, edge_lengths):
        if len(vehicle_list) == 0:
            return {}

        total_demand = len(vehicle_list)
        allocation_results = {}
        self.congestion_tracker.step_forward()
        self._step_counter += 1


        if self._step_counter % self._congestion_update_interval == 0 or self._cached_risk is None:
            congestion_risk = self.calculate_congestion_risk()
            congested_mask = self.congestion_tracker.get_congested_mask(congestion_risk)
            self._cached_risk = congestion_risk
            self._cached_mask = congested_mask
        else:
            congestion_risk = self._cached_risk
            congested_mask = self._cached_mask

        od_groups = defaultdict(list)
        for veh in vehicle_list:
            veh_id, from_e, to_e, _ = veh
            od_groups[(from_e, to_e)].append((veh_id, from_e, to_e))

        all_batch_edges = []
        all_batch_offsets = []

        for od_key, vehs in od_groups.items():
            if od_key not in self.od_candidate_paths:
                for vid, fe, te in vehs:
                    allocation_results[vid] = None
                continue

            paths = self._path_edges_cache[od_key]
            num_paths = len(paths)
            path_weights = self._path_priority_cache.get(od_key, np.zeros(num_paths, dtype=np.float32))

            single_limit = self.congestion_tracker.get_single_edge_limit(len(vehs))
            max_alloc_total = min(self.max_allocation_ratio * len(vehs), single_limit * num_paths)
            path_allocation = _numba_allocate_paths_core(num_paths, len(vehs), max_alloc_total, path_weights)

            flat_edges, _ = self._path_congest_cache[od_key]
            congested_flags = congested_mask[flat_edges]
            reduce_idx = self._path_reduce_idx_cache[od_key]
            path_has_congest = np.logical_or.reduceat(congested_flags, reduce_idx)

            high_congest_idx = np.where(path_has_congest)[0]
            normal_idx = np.where(~path_has_congest)[0]
            global_limit = self.congestion_tracker.get_od_congested_limit(len(vehs))

            if len(high_congest_idx) > 0 and self.feedback_gain > 1e-6:
                path_avg_risk = np.add.reduceat(congestion_risk[flat_edges], reduce_idx) / np.add.reduceat(
                    np.ones_like(congested_flags), reduce_idx)
                congest_risk = path_avg_risk[high_congest_idx]
                risk_sum = np.sum(congest_risk)
                if risk_sum > 1e-6:
                    risk_share = congest_risk / risk_sum
                else:
                    risk_share = np.full(len(high_congest_idx), 1.0 / len(high_congest_idx), dtype=np.float32)

                normal_weights = path_weights[normal_idx] if len(normal_idx) > 0 else np.zeros(0, dtype=np.float32)
                normal_weight_sum = np.sum(normal_weights)

                unmet_reduction = 0.0
                for _ in range(num_paths + 1):
                    high_congest_alloc = np.sum(path_allocation[high_congest_idx])
                    if high_congest_alloc <= global_limit + 1e-6:
                        break

                    excess = high_congest_alloc - global_limit
                    reduction = np.clip(self.feedback_gain * excess, 0.0, excess)
                    if reduction <= 1e-6:
                        break

                    desired_cut = reduction * risk_share
                    actual_cut = np.minimum(desired_cut, path_allocation[high_congest_idx])
                    path_allocation[high_congest_idx] -= actual_cut
                    applied = float(np.sum(actual_cut))
                    if applied <= 1e-6:
                        unmet_reduction += reduction
                        break

                    if len(normal_idx) > 0:
                        room = np.maximum(max_alloc_total - path_allocation[normal_idx], 0.0)
                        if normal_weight_sum > 1e-6:
                            desired_add = applied * normal_weights / normal_weight_sum
                        else:
                            desired_add = np.full(len(normal_idx), applied / len(normal_idx), dtype=np.float32)
                        actual_add = np.minimum(desired_add, room)
                        path_allocation[normal_idx] += actual_add
                        placed = float(np.sum(actual_add))
                    else:
                        inverse_risk = 1.0 / np.maximum(congest_risk, 1e-6)
                        inverse_risk_sum = np.sum(inverse_risk)
                        if inverse_risk_sum > 1e-6:
                            path_allocation[high_congest_idx] += applied * inverse_risk / inverse_risk_sum
                        else:
                            path_allocation[high_congest_idx] += applied / len(high_congest_idx)
                        placed = applied

                    leftover = applied - placed
                    if leftover > 1e-6:
                        path_allocation[high_congest_idx] += leftover * risk_share

                if unmet_reduction > 1e-6:
                    self.unmet_demand_total += unmet_reduction

            n_vehs = len(vehs)
            alloc_cumsum = np.cumsum(path_allocation)
            veh_indices = np.arange(n_vehs, dtype=np.float32) + 0.5
            path_idx_arr = np.searchsorted(alloc_cumsum, veh_indices)
            path_idx_arr = np.clip(path_idx_arr, 0, num_paths - 1)

            for idx, (vid, fe, te) in enumerate(vehs):
                allocation_results[vid] = int(path_idx_arr[idx])

            path_edges = self._path_edges_cache[od_key]
            path_offsets = self._path_offsets_cache[od_key]
            for p_idx in path_idx_arr:
                all_batch_edges.append(path_edges[p_idx])
                all_batch_offsets.append(path_offsets[p_idx])

        if all_batch_edges:
            all_edges = np.concatenate(all_batch_edges)
            all_offsets = np.concatenate(all_batch_offsets)
            self.congestion_tracker.update_allocation(all_edges, all_offsets)

        self.total_demand += total_demand
        self.total_count += 1
        self.history_congestion_ratio[:, self.window_idx] = congestion_risk
        self.window_idx = (self.window_idx + 1) % self.smooth_window

        return allocation_results

    def calculate_congestion_risk(self):

        return self.congestion_tracker.calculate_congestion_risk()

    def reset(self):
        self.allocate_count[:] = 0
        self.cum_usage[:] = 0
        self.total_count = 0
        self.total_demand = 0.0
        self.unmet_demand_total = 0.0
        self.edge_status[:] = True
        self.history_congestion_ratio[:] = 0
        self.window_idx = 0
        self._step_counter = 0
        self._cached_risk = None
        self._cached_mask = None
        self.congestion_tracker = EndogenousCongestionTracker(
            num_edges=self.num_edges,
            congestion_threshold=self.congestion_tracker.congestion_threshold,
            max_single_edge_ratio=self.congestion_tracker.max_single_edge_ratio,
            min_base_allocation=self.congestion_tracker.min_base_allocation,
            time_window_size=self.smooth_window,
            decay_factor=self.congestion_tracker.decay_factor,
            gamma_max=self.congestion_tracker.max_od_congested_ratio
        )

        self._path_priority_cache.clear()
        self._path_edge_ids_cache.clear()
        self._path_edge_pos_cache.clear()
        self._path_edges_cache.clear()
        self._path_offsets_cache.clear()
        self._path_congest_cache.clear()
        self._path_reduce_idx_cache.clear()



class D3STN_EnhancedTracker(EndogenousCongestionTracker):
    def __init__(self, num_edges,
                 enable_dynamic_tau=True,
                 **kwargs):
        super().__init__(num_edges, **kwargs)

        self.enable_dynamic_tau = enable_dynamic_tau

        self._base_decay_pows = self._decay_pows.copy()

        self.dynamic_tau = self.decay_factor
        self.tau_learning_rate = 0.001
        self.prediction_error_history = []
        self.last_prediction = np.ones(num_edges, dtype=np.float32) * self.min_base_allocation

        self.hybrid_adj = None
        self.gnn_layers = 2
        self._adj_built = False

        self.num_heads = 3
        self.kernel_size = 5
        self.attention_weights = None
        self._init_conv_kernels()

    def _init_conv_kernels(self):
        self.conv_kernels = []
        for _ in range(self.num_heads):
            kernel = np.exp(-np.linspace(-2, 2, self.kernel_size) ** 2 / 2)
            kernel = kernel / kernel.sum()
            self.conv_kernels.append(kernel.astype(np.float32))

    def build_hybrid_adjacency(self, adjacency_matrix, edge_lengths, free_flow_speeds):
        num = self.num_edges
        A_road_sparse = sp.csr_matrix(adjacency_matrix.astype(np.float32))

        free_flow_times = edge_lengths / np.maximum(free_flow_speeds, 1e-3)
        ft_max = free_flow_times.max()
        ft_min = free_flow_times.min()
        time_span = ft_max - ft_min
        max_diff = time_span if time_span > 0 else 1

        A_road_plus_I = (A_road_sparse.astype(np.float64) + sp.eye(num, format='csr', dtype=np.float64)).tocoo()
        rows = A_road_plus_I.row
        cols = A_road_plus_I.col
        support_vals = A_road_plus_I.data

        time_diff_support = np.abs(free_flow_times[rows] - free_flow_times[cols])
        A_func_support_vals = (1.0 - time_diff_support / max_diff) * support_vals
        A_func_masked = sp.csr_matrix((A_func_support_vals, (rows, cols)), shape=(num, num))

        hybrid_adj = (A_road_sparse.multiply(0.6) + A_func_masked.multiply(0.4)).tocsr()

        row_sum = np.asarray(hybrid_adj.sum(axis=1)).flatten()
        safe_mask = row_sum > 1e-6

        row_of_nonzero = np.repeat(np.arange(num), np.diff(hybrid_adj.indptr))
        new_data = np.zeros_like(hybrid_adj.data)
        safe_nonzero_mask = safe_mask[row_of_nonzero]
        new_data[safe_nonzero_mask] = hybrid_adj.data[safe_nonzero_mask] / row_sum[row_of_nonzero[safe_nonzero_mask]]
        hybrid_adj.data = new_data

        self.hybrid_adj = hybrid_adj
        self._adj_built = True

    def _conv_attention_aggregate(self, window_data):
        num_edges, window_len = window_data.shape
        head_outputs = []
        kernel_size = self.conv_kernels[0].shape[0]
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size - 1 - pad_left

        for head_idx in range(self.num_heads):
            kernel = self.conv_kernels[head_idx]
            padded = np.pad(window_data, ((0, 0), (pad_left, pad_right)), mode='constant')
            windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size, axis=1)
            conv_result = windows @ kernel[::-1]

            time_weights = np.linspace(0.5, 1.5, window_len).astype(np.float32)
            if head_idx == 0:
                time_weights = np.exp(np.linspace(0, 1, window_len)).astype(np.float32)
            elif head_idx == 1:
                time_weights = np.ones(window_len, dtype=np.float32)
            else:
                time_weights = np.exp(np.linspace(1, 0, window_len)).astype(np.float32)

            time_weights = time_weights / time_weights.sum()
            weighted = conv_result * time_weights[None, :]
            head_outputs.append(weighted.sum(axis=1))

        aggregated = np.mean(head_outputs, axis=0)
        return aggregated

    def _gnn_spatial_propagate(self, node_features):
        if not self._adj_built:
            return node_features

        x = node_features.copy()
        for _ in range(self.gnn_layers):
            x = self.hybrid_adj @ x
            x = np.maximum(x, 0)
        return x

    def _dynamic_tau_adjust(self, current_risk):

        if not self.enable_dynamic_tau:
            return

        error = np.mean(np.abs(current_risk - self.last_prediction))
        self.prediction_error_history.append(error)
        if len(self.prediction_error_history) > 30:
            self.prediction_error_history.pop(0)

        if len(self.prediction_error_history) >= 10:
            error_trend = np.mean(self.prediction_error_history[-5:]) - np.mean(self.prediction_error_history[-10:-5])
            tau_delta = error_trend * self.tau_learning_rate
            self.dynamic_tau = np.clip(self.dynamic_tau - tau_delta, 0.8, 0.99)
            self._decay_pows = self.dynamic_tau ** np.arange(self.time_window_size, dtype=np.float32)

        self.last_prediction = current_risk.copy()

    def calculate_congestion_risk(self):

        if self.enable_dynamic_tau:
            window_total = np.sum(self.allocation_windows * self._decay_pows[None, :], axis=1)
        else:
            window_total = np.sum(self.allocation_windows * self._base_decay_pows[None, :], axis=1)

        attn_total = self._conv_attention_aggregate(self.allocation_windows)
        spatial_enhanced = self._gnn_spatial_propagate(attn_total)
        fused_risk = 0.5 * window_total + 0.5 * spatial_enhanced

        congestion_risk = np.divide(fused_risk, self.edge_capacity,
                                    out=np.zeros_like(fused_risk),
                                    where=self.edge_capacity > 1e-6)
        congestion_risk = np.maximum(congestion_risk, self.min_base_allocation)

        self._dynamic_tau_adjust(congestion_risk)
        return congestion_risk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import diags
from scipy.linalg import eigh



torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True





def calculate_normalized_laplacian(adj):
    n = adj.shape[0]
    adj_sparse = adj.tocsr() if sp.issparse(adj) else sp.csr_matrix(adj)
    degree = np.asarray(adj_sparse.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5, where=degree!=0)
    D_inv_sqrt = diags(degree_inv_sqrt)
    scaled_adj = (D_inv_sqrt @ adj_sparse @ D_inv_sqrt).tocoo()
    return scaled_adj, n


def chebyshev_polynomials(scaled_adj_coo, n, K, device):
    identity_gpu = torch.eye(n, device=device)
    if K <= 1:
        result = identity_gpu.unsqueeze(0).clone().float()
        del identity_gpu
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return result

    indices = torch.from_numpy(np.vstack((scaled_adj_coo.row, scaled_adj_coo.col))).long()
    values = torch.from_numpy(scaled_adj_coo.data).float()
    scaled_adj_gpu = torch.sparse_coo_tensor(indices, values, torch.Size((n, n))).coalesce().to(device)
    laplacian_gpu = identity_gpu - scaled_adj_gpu.to_dense()

    lambda_max = torch.linalg.eigvalsh(laplacian_gpu)[-1]
    scaled_laplacian = (2.0 / lambda_max) * laplacian_gpu - identity_gpu

    cheb_polys = [identity_gpu, scaled_laplacian]
    for k in range(2, K):
        cheb_polys.append(2 * scaled_laplacian @ cheb_polys[k - 1] - cheb_polys[k - 2])

    result = torch.stack(cheb_polys[:K]).float()
    del indices, values, scaled_adj_gpu, laplacian_gpu, lambda_max, scaled_laplacian, cheb_polys, identity_gpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result






class ChebGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, cheb_polys):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.K = K

        self.register_buffer('cheb_polys', cheb_polys)
        self.weight = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        B, _, N, T = x.shape
        x_reshaped = x.permute(0, 3, 1, 2).reshape(B * T, self.in_c, N)


        cheb_out = torch.einsum('bcn,knm->bckm', x_reshaped, self.cheb_polys)
        output = torch.einsum('bckm,kci->bmi', cheb_out, self.weight)

        output = output + self.bias[None, :]
        output = output.reshape(B, T, self.out_c, N).permute(0, 2, 3, 1)
        return output


class FirstOrderGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels

        n = adj.shape[0]
        adj_tilde = adj + np.eye(n)
        degree_tilde = adj_tilde.sum(axis=1)
        degree_inv_sqrt = np.power(degree_tilde, -0.5, where=degree_tilde != 0)
        degree_mat_inv_sqrt = diags(degree_inv_sqrt).toarray()
        adj_hat = degree_mat_inv_sqrt @ adj_tilde @ degree_mat_inv_sqrt

        self.register_buffer('adj_hat', torch.from_numpy(adj_hat).float())
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        B, _, N, T = x.shape
        x_reshaped = x.permute(0, 3, 1, 2).reshape(B * T, self.in_c, N)

        x_linear = x_reshaped.transpose(1, 2) @ self.weight
        x_conv = self.adj_hat @ x_linear
        x_conv = x_conv + self.bias[None, :]

        output = x_conv.transpose(1, 2).reshape(B, T, self.out_c, N).permute(0, 2, 3, 1)
        return output





class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, kernel_size))
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out = self.conv(x)
        P, Q = out.chunk(2, dim=1)
        return P * torch.sigmoid(Q)





class STConvBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, kernel_size,
                 graph_conv_type, graph_conv_params):
        super().__init__()
        self.temporal_conv1 = TemporalGatedConv(in_channels, out_channels, kernel_size)

        if graph_conv_type == 'cheb':
            self.graph_conv = ChebGraphConv(out_channels, spatial_channels, *graph_conv_params)
        elif graph_conv_type == 'first_order':
            self.graph_conv = FirstOrderGraphConv(out_channels, spatial_channels, *graph_conv_params)
        else:
            raise ValueError("仅支持 cheb/first_order")

        self.temporal_conv2 = TemporalGatedConv(spatial_channels, out_channels, kernel_size)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.temporal_conv1(x)
        x = F.relu(self.graph_conv(x))
        x = self.temporal_conv2(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x






class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, spatial_channels, out_channels, kernel_size,
                 seq_len, pred_steps, graph_conv_type, graph_conv_params,
                 scaler_mean=None, scaler_std=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_c = out_channels
        self.pred_steps = pred_steps


        self.st_block1 = STConvBlock(in_channels, spatial_channels, out_channels, num_nodes,
                                     kernel_size, graph_conv_type, graph_conv_params)
        self.st_block2 = STConvBlock(out_channels, spatial_channels, out_channels, num_nodes,
                                     kernel_size, graph_conv_type, graph_conv_params)

        self.final_seq_len = seq_len - 4 * (kernel_size - 1)
        self.output_fc = nn.Linear(out_channels * self.final_seq_len, pred_steps)



        self.register_buffer('scaler_mean', torch.tensor(scaler_mean, dtype=torch.float32) if scaler_mean is not None else None)
        self.register_buffer('scaler_std', torch.tensor(scaler_std, dtype=torch.float32) if scaler_std is not None else None)

    def forward(self, x):

        x = self.st_block1(x)
        x = self.st_block2(x)
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], self.num_nodes, -1)
        output = self.output_fc(x)



        if self.scaler_mean is not None and self.scaler_std is not None:
            output = output * self.scaler_std + self.scaler_mean


        output = torch.clamp(output, min=0.0)
        return output


    def save_model(self, save_path):
        save_dict = {
            "state_dict": self.state_dict(),
            "scaler_mean": self.scaler_mean.item() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.item() if self.scaler_std is not None else None
        }
        torch.save(save_dict, save_path)

    def load_model(self, load_path, device='cpu'):
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for key, param in state_dict.items():
            if 'temporal_conv' in key and 'conv.weight' in key and param.dim() == 3:
                new_state_dict[key] = param.unsqueeze(2)
            elif 'graph_conv.weight' in key:
                if param.shape[0] != self.st_block1.graph_conv.weight.shape[0]:
                    print(f"Warning: {key} shape mismatch (checkpoint: {param.shape}, model: {self.st_block1.graph_conv.weight.shape}), using model's K")
                    continue
            else:
                new_state_dict[key] = param

        self.load_state_dict(new_state_dict, strict=False)

        if checkpoint.get("scaler_mean") is not None:
            self.scaler_mean = torch.tensor(checkpoint["scaler_mean"], dtype=torch.float32, device=device)
            self.scaler_std = torch.tensor(checkpoint["scaler_std"], dtype=torch.float32, device=device)
        self.eval()





class ZScoreScaler_stgcn:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, dtype=np.float32)
        self.std = np.std(data, dtype=np.float32)

        if self.std < 1e-6:
            self.std = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def create_sliding_window_dataset(data, seq_len, pred_len):
    if data.ndim == 2:
        data = data[..., None]
    T, N, F = data.shape
    indices = np.arange(T - seq_len - pred_len + 1)[:, None] + np.arange(seq_len + pred_len)
    samples = data[indices]
    X = samples[:, :seq_len].astype(np.float32)
    Y = samples[:, seq_len:, :, 0].transpose(0, 2, 1).astype(np.float32)
    return X, Y

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import diags, csr_matrix
from torch.utils.data import Dataset


torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True





def build_adj_matrix(adjacency_matrix):
    adj_sparse = adjacency_matrix.tocsr() if sp.issparse(adjacency_matrix) else sp.csr_matrix(adjacency_matrix)
    adj_sparse = adj_sparse.astype(np.float64)
    adj_sparse.setdiag(0.0)
    adj_sparse.eliminate_zeros()
    return adj_sparse


def compute_diffusion_matrices(adj, K):
    out_degree = np.array(adj.sum(axis=1)).flatten()

    out_degree_safe = np.where(out_degree > 0, out_degree, 1.0)
    out_degree_inv = np.where(out_degree > 0, 1.0 / out_degree_safe, 0.0)
    forward_trans = diags(out_degree_inv) @ adj

    in_degree = np.array(adj.sum(axis=0)).flatten()
    in_degree_safe = np.where(in_degree > 0, in_degree, 1.0)
    in_degree_inv = np.where(in_degree > 0, 1.0 / in_degree_safe, 0.0)
    backward_trans = diags(in_degree_inv) @ adj.T

    n = adj.shape[0]
    forward_dense = np.asarray(forward_trans.todense())
    backward_dense = np.asarray(backward_trans.todense())

    forward_polys = [np.eye(n)]
    for k in range(1, K):
        forward_polys.append(forward_polys[-1] @ forward_dense)

    backward_polys = [np.eye(n)]
    for k in range(1, K):
        backward_polys.append(backward_polys[-1] @ backward_dense)

    forward_stack = torch.from_numpy(np.stack(forward_polys, axis=0)).float()
    backward_stack = torch.from_numpy(np.stack(backward_polys, axis=0)).float()

    return forward_stack, backward_stack


def scheduled_sampling_prob(iteration, tau=3000):
    return tau / (tau + np.exp(iteration / tau))





class DiffusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, forward_trans, backward_trans, num_nodes):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.K = K
        self.num_nodes = num_nodes

        self.register_buffer('forward_trans', forward_trans.float())
        self.register_buffer('backward_trans', backward_trans.float())

        self.theta = nn.Parameter(torch.FloatTensor(out_channels, in_channels, K, 2))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        x_reshaped = x.reshape(batch_size * self.in_c, self.num_nodes)

        forward_supports = []
        backward_supports = []

        x_f = x_reshaped
        x_b = x_reshaped
        for k in range(self.K):
            forward_supports.append(x_f.reshape(batch_size, self.in_c, self.num_nodes))
            backward_supports.append(x_b.reshape(batch_size, self.in_c, self.num_nodes))
            if k < self.K - 1:
                x_f = torch.matmul(x_f, self.forward_trans[k])
                x_b = torch.matmul(x_b, self.backward_trans[k])

        out = torch.zeros(batch_size, self.out_c, self.num_nodes, device=x.device)
        for k in range(self.K):
            out += torch.einsum('oc,bcn->bon', self.theta[:, :, k, 0], forward_supports[k])
            out += torch.einsum('oc,bcn->bon', self.theta[:, :, k, 1], backward_supports[k])

        out = out + self.bias[:, None]
        return out.permute(0, 2, 1)


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, forward_trans, backward_trans, num_nodes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_conv = DiffusionConv(input_dim + hidden_dim, 2 * hidden_dim, K, forward_trans, backward_trans, num_nodes)
        self.candidate_conv = DiffusionConv(input_dim + hidden_dim, hidden_dim, K, forward_trans, backward_trans, num_nodes)

    def forward(self, x_t, h_prev):
        concat_gate = torch.cat([x_t, h_prev], dim=-1)
        r_t, u_t = torch.chunk(self.gate_conv(concat_gate), 2, dim=-1)
        r_t, u_t = torch.sigmoid(r_t), torch.sigmoid(u_t)

        concat_cand = torch.cat([x_t, r_t * h_prev], dim=-1)
        c_t = torch.tanh(self.candidate_conv(concat_cand))

        return u_t * h_prev + (1 - u_t) * c_t


class DCGRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, forward_trans, backward_trans, num_nodes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = DCGRUCell(input_dim, hidden_dim, K, forward_trans, backward_trans, num_nodes)

    def forward(self, x_seq, h0=None):
        batch_size, seq_len, num_nodes, _ = x_seq.shape
        h0 = h0 if h0 is not None else torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x_seq.device)

        h = h0
        outputs = torch.empty(seq_len, batch_size, num_nodes, self.hidden_dim, device=x_seq.device)
        x_seq = x_seq.permute(1, 0, 2, 3).contiguous()

        for t in range(seq_len):
            h = self.cell(x_seq[t], h)
            outputs[t] = h

        return outputs.permute(1, 0, 2, 3).contiguous(), h





class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, K, forward_trans, backward_trans, num_nodes):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            DCGRULayer(input_dim if i == 0 else hidden_dim, hidden_dim, K, forward_trans, backward_trans, num_nodes)
            for i in range(num_layers)
        ])

    def forward(self, x_seq):
        h_n, current_input = [], x_seq
        for layer in self.layers:
            current_input, h = layer(current_input)
            h_n.append(h)
        return current_input, torch.stack(h_n, dim=0)


class DCRNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, K, forward_trans, backward_trans, num_nodes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            DCGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, K, forward_trans, backward_trans, num_nodes)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, h0, targets=None, pred_len=12, use_sampling=False, sampling_prob=1.0):
        batch_size, num_nodes = h0.shape[1], h0.shape[2]
        device = h0.device

        x_t = torch.zeros(batch_size, num_nodes, self.output_dim, device=device)
        h = [h0[i] for i in range(self.num_layers)]

        predictions = torch.empty(pred_len, batch_size, num_nodes, self.output_dim, device=device)

        for t in range(pred_len):
            current_input = x_t
            for i in range(self.num_layers):
                h[i] = self.layers[i](current_input, h[i])
                current_input = h[i]

            pred_t = self.output_proj(current_input)
            predictions[t] = pred_t

            if use_sampling and targets is not None:
                x_t = targets[:, t, :, :] if torch.rand(1).item() < sampling_prob else pred_t
            else:
                x_t = pred_t

        return predictions.permute(1, 0, 2, 3)





class DCRNN(nn.Module):
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=1,
                 seq_len=12, pred_len=12, K=3, num_layers=2,
                 forward_trans=None, backward_trans=None,
                 scaler_mean=None, scaler_std=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes

        self.encoder = DCRNNEncoder(input_dim, hidden_dim, num_layers, K, forward_trans, backward_trans, num_nodes)
        self.decoder = DCRNNDecoder(output_dim, hidden_dim, output_dim, num_layers, K, forward_trans, backward_trans, num_nodes)

        self.register_buffer(
            'scaler_mean',
            torch.tensor(scaler_mean, dtype=torch.float32) if scaler_mean is not None else None
        )
        self.register_buffer(
            'scaler_std',
            torch.tensor(scaler_std, dtype=torch.float32) if scaler_std is not None else None
        )

    def forward(self, x, targets=None, training=False, sampling_prob=1.0):
        _, encoder_hidden = self.encoder(x)
        output = self.decoder(encoder_hidden, targets, self.pred_len, training, sampling_prob)

        if self.scaler_mean is not None and self.scaler_std is not None:
            output = output * self.scaler_std + self.scaler_mean

        output = torch.clamp(output, min=0.0, max=1.0)
        return output

    def save_model(self, save_path):
        save_dict = {
            "state_dict": self.state_dict(),
            "scaler_mean": self.scaler_mean.item() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.item() if self.scaler_std is not None else None
        }
        torch.save(save_dict, save_path)

    def load_model(self, load_path, device='cpu'):
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint["state_dict"]

        filtered_state_dict = {}
        for key, param in state_dict.items():
            if "forward_trans" not in key and "backward_trans" not in key:
                if key in self.state_dict():
                    model_param = self.state_dict()[key]
                    if param.shape == model_param.shape:
                        filtered_state_dict[key] = param
                    else:
                        print(f"Warning: {key} shape mismatch (checkpoint: {param.shape}, model: {model_param.shape}), skipping")
                else:
                    filtered_state_dict[key] = param

        self.load_state_dict(filtered_state_dict, strict=False)

        if checkpoint.get("scaler_mean") is not None:
            self.scaler_mean = torch.tensor(checkpoint["scaler_mean"], dtype=torch.float32, device=device)
            self.scaler_std = torch.tensor(checkpoint["scaler_std"], dtype=torch.float32, device=device)
        self.eval()





class ZScoreScaler_dcrnn:
    def __init__(self):
        self.mean = self.std = None

    def fit(self, data):
        self.mean = np.mean(data, dtype=np.float32)
        self.std = np.std(data, dtype=np.float32)
        if self.std < 1e-6:
            self.std = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class SlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = len(data) - seq_len - pred_len + 1
        if self.total_len <= 0:
            raise ValueError(f"数据长度不足：seq_len={seq_len}, pred_len={pred_len}, data_len={len(data)}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len, :]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len, :]
        return x.astype(np.float32), y.astype(np.float32)


def create_sliding_window_dataset_dcrnn(data, seq_len, pred_len):
    return SlidingWindowDataset(data, seq_len, pred_len)



def masked_mae(pred, true, null_val=0.0):
    mask = (true != null_val).float()
    return torch.sum(torch.abs(pred - true) * mask) / torch.sum(mask)


def masked_rmse(pred, true, null_val=0.0):
    mask = (true != null_val).float()
    return torch.sqrt(torch.sum(torch.square(pred - true) * mask) / torch.sum(mask))


def masked_mape(pred, true, null_val=0.0):
    mask = (true != null_val).float()
    return torch.sum(torch.abs((pred - true) / (true + 1e-8)) * mask) / torch.sum(mask)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True





class ResidualUnit(nn.Module):
    def __init__(self, channels: int, use_bn: bool = True):
        super().__init__()
        layers = []
        layers.append(nn.BatchNorm2d(channels) if use_bn else nn.Identity())
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(channels) if use_bn else nn.Identity())
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True))
        self.residual_func = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_func(x)


class ResidualBranch(nn.Module):
    def __init__(self, in_channels: int, num_res_units: int, use_bn: bool = True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[
            ResidualUnit(64, use_bn) for _ in range(num_res_units)
        ])

        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size, seq_len * channels, height, width)
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.conv2(x)
        return x


class ExternalModule(nn.Module):
    def __init__(self, ext_dim: int, output_dim: int):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(ext_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(x)





class STResNet(nn.Module):
    def __init__(
            self,
            height: int,
            width: int,
            closeness_len: int,
            period_len: int,
            trend_len: int,
            num_res_units: int = 12,
            use_bn: bool = True,
            ext_dim: int = None,
            scaler_min: float = None,
            scaler_max: float = None
    ):
        super().__init__()
        self.height = height
        self.width = width

        self.spatial_dim = 1 * height * width


        self.closeness_net = ResidualBranch(1 * closeness_len, num_res_units, use_bn)
        self.period_net = ResidualBranch(1 * period_len, num_res_units, use_bn)
        self.trend_net = ResidualBranch(1 * trend_len, num_res_units, use_bn)


        self.W_c = nn.Parameter(torch.full((1, height, width), 1 / 3))
        self.W_p = nn.Parameter(torch.full((1, height, width), 1 / 3))
        self.W_q = nn.Parameter(torch.full((1, height, width), 1 / 3))

        self.use_external = ext_dim is not None
        if self.use_external:
            self.external_net = ExternalModule(ext_dim, self.spatial_dim)

        self.final_activation = nn.Tanh()

        self.register_buffer(
            'scaler_min',
            torch.tensor(scaler_min, dtype=torch.float32) if scaler_min is not None else None
        )
        self.register_buffer(
            'scaler_max',
            torch.tensor(scaler_max, dtype=torch.float32) if scaler_max is not None else None
        )

    def forward(
            self,
            closeness_x: torch.Tensor,
            period_x: torch.Tensor,
            trend_x: torch.Tensor,
            ext_x: torch.Tensor = None,
            return_original: bool = False
    ) -> torch.Tensor:
        c_out = self.closeness_net(closeness_x)
        p_out = self.period_net(period_x)
        q_out = self.trend_net(trend_x)

        fused_res = self.W_c * c_out + self.W_p * p_out + self.W_q * q_out

        if self.use_external and ext_x is not None:
            ext_out = self.external_net(ext_x)
            ext_out = ext_out.view(-1, 1, self.height, self.width)
            fused_res = fused_res + ext_out

        output = self.final_activation(fused_res)


        if return_original and self.scaler_min is not None and self.scaler_max is not None:
            output = self.scaler_min + (output - (-1)) * (self.scaler_max - self.scaler_min) / 2
            output = torch.clamp(output, min=0.0)

        return output

    def save_model(self, save_path):
        save_dict = {
            "state_dict": self.state_dict(),
            "scaler_min": self.scaler_min.item() if self.scaler_min is not None else None,
            "scaler_max": self.scaler_max.item() if self.scaler_max is not None else None
        }
        torch.save(save_dict, save_path)

    def load_model(self, load_path, device='cpu'):
        import torch.nn.functional as F
        checkpoint = torch.load(load_path, map_location=device)
        ckpt_state_dict = checkpoint["state_dict"]

        model_state_dict = self.state_dict()
        filtered_state_dict = {}
        mismatched_keys = []
        adapted_keys = []

        for key, ckpt_param in ckpt_state_dict.items():
            if key not in model_state_dict:
                continue

            model_param = model_state_dict[key]
            if ckpt_param.shape == model_param.shape:

                filtered_state_dict[key] = ckpt_param

            elif key in ("W_c", "W_p", "W_q") and ckpt_param.dim() == 3 and model_param.dim() == 3:

                ckpt_4d = ckpt_param.unsqueeze(0)
                target_size = (model_param.shape[1], model_param.shape[2])

                adapted_param = F.interpolate(
                    ckpt_4d, size=target_size, mode='bilinear', align_corners=True
                ).squeeze(0)
                filtered_state_dict[key] = adapted_param
                adapted_keys.append((
                    key,
                    tuple(ckpt_param.shape),
                    tuple(model_param.shape)
                ))
            else:

                mismatched_keys.append((
                    key,
                    tuple(ckpt_param.shape),
                    tuple(model_param.shape)
                ))


        self.load_state_dict(filtered_state_dict, strict=False)


        if adapted_keys:
            print("[INFO] 以下空间融合权重已通过插值适配到当前路网尺寸，成功加载：")
            for k, ckpt_shape, model_shape in adapted_keys:
                print(f"  - {k}: 原尺寸{ckpt_shape} → 当前尺寸{model_shape}")


        if mismatched_keys:
            print("[WARNING] 以下参数因形状不匹配，未从预训练权重加载，将使用初始化值：")
            for k, ckpt_shape, model_shape in mismatched_keys:
                print(f"  - {k}: 权重形状{ckpt_shape}，当前模型形状{model_shape}")


        if checkpoint.get("scaler_min") is not None:
            self.scaler_min = torch.tensor(checkpoint["scaler_min"], dtype=torch.float32, device=device)
            self.scaler_max = torch.tensor(checkpoint["scaler_max"], dtype=torch.float32, device=device)
        self.eval()





class MinMaxScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.min_val = None
        self.max_val = None
        self.range_min, self.range_max = feature_range
        self.scale = None

    def fit(self, data: np.ndarray):
        self.min_val = np.min(data)
        self.max_val = np.max(data)
        if abs(self.max_val - self.min_val) < 1e-8:
            self.scale = 1.0
        else:
            self.scale = (self.range_max - self.range_min) / (self.max_val - self.min_val)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.range_min + (data - self.min_val) * self.scale

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if abs(self.scale) < 1e-8:
            return data
        return self.min_val + (data - self.range_min) / self.scale


class STResNetDataset(Dataset):
    def __init__(self, flow_data, closeness_len, period_len, trend_len, period_interval, trend_interval):
        self.flow_data = flow_data
        self.closeness_len = closeness_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.period_interval = period_interval
        self.trend_interval = trend_interval
        self.max_lag = max(closeness_len, period_len * period_interval, trend_len * trend_interval)
        self.total_len = len(flow_data) - self.max_lag
        if self.total_len <= 0:
            raise ValueError("数据长度不足，无法构建窗口")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        t = idx + self.max_lag

        c = self.flow_data[t - self.closeness_len : t]
        c = c.transpose(1, 0, 2, 3).reshape(-1, self.flow_data.shape[2], self.flow_data.shape[3])

        p_indices = t - np.arange(1, self.period_len + 1)[::-1] * self.period_interval
        p = self.flow_data[p_indices]
        p = p.transpose(1, 0, 2, 3).reshape(-1, self.flow_data.shape[2], self.flow_data.shape[3])

        q_indices = t - np.arange(1, self.trend_len + 1)[::-1] * self.trend_interval
        q = self.flow_data[q_indices]
        q = q.transpose(1, 0, 2, 3).reshape(-1, self.flow_data.shape[2], self.flow_data.shape[3])

        y = self.flow_data[t]
        return c.astype(np.float32), p.astype(np.float32), q.astype(np.float32), y.astype(np.float32)


def build_stresnet_dataset(flow_data, closeness_len, period_len, trend_len, period_interval, trend_interval, ext_data=None):
    return STResNetDataset(flow_data, closeness_len, period_len, trend_len, period_interval, trend_interval)




import numpy as np

class CongestionPredictionEvaluator:
    def __init__(self, num_edges,
                 business_threshold=0.4,
                 top_ratio=0.1,
                 window_size=60,
                 real_congestion_threshold=None):
        self.num_edges = num_edges

        if np.isscalar(business_threshold):
            self.business_threshold = np.full(num_edges, business_threshold, dtype=np.float32)
        else:
            self.business_threshold = np.asarray(business_threshold, dtype=np.float32)



        self.real_congestion_threshold = (
            real_congestion_threshold if real_congestion_threshold is not None
            else float(np.mean(self.business_threshold))
        )

        self.top_ratio = top_ratio
        self.window_size = window_size

        self.pred_risk_history = []
        self.real_occupancy_history = []
        self._pred_quantile_threshold = None

    def record_step(self, pred_risk, real_occupancy):
        self.pred_risk_history.append(pred_risk.copy())
        self.real_occupancy_history.append(real_occupancy.copy())

    def _compute_pred_based_threshold(self):
        all_pred = np.concatenate(self.pred_risk_history)
        self._pred_quantile_threshold = np.quantile(all_pred, 1 - self.top_ratio)
        return self._pred_quantile_threshold

    def _sliding_mean(self, data):
        if len(data) < self.window_size:
            return data
        cumsum = np.cumsum(data, axis=0, dtype=np.float64)
        cumsum = np.concatenate([np.zeros((1, data.shape[1]), dtype=np.float64), cumsum], axis=0)
        idx = np.arange(len(data))
        start_idx = np.maximum(idx - self.window_size + 1, 0)
        counts = (idx - start_idx + 1).astype(np.float64)
        window_sum = cumsum[idx + 1] - cumsum[start_idx]
        smoothed = window_sum / counts[:, None]
        return smoothed.astype(data.dtype)

    def calculate_metrics(self):
        pred_risks = np.array(self.pred_risk_history)
        real_occs = np.array(self.real_occupancy_history)
        real_smoothed = self._sliding_mean(real_occs)
        pred_quant_thresh = self._compute_pred_based_threshold()


        lead_time_full = np.full(self.num_edges, np.nan, dtype=np.float64)
        miss_flags = np.zeros(self.num_edges, dtype=bool)
        late_flags = np.zeros(self.num_edges, dtype=bool)
        pre_intervention_count = 0
        real_congested_edge_count = 0
        business_mask_full = pred_risks >= self.business_threshold
        real_cong_mask_thresh = real_smoothed >= self.real_congestion_threshold

        for e_idx in range(self.num_edges):
            pred_series = pred_risks[:, e_idx]
            real_series = real_smoothed[:, e_idx]


            real_first_idx = np.where(real_series >= self.real_congestion_threshold)[0]

            pred_first_idx = np.where(pred_series >= self.business_threshold[e_idx])[0]

            if len(real_first_idx) > 0:
                real_congested_edge_count += 1

                if len(pred_first_idx) > 0 and pred_first_idx[0] <= real_first_idx[0]:
                    pre_intervention_count += 1

                    lead = real_first_idx[0] - pred_first_idx[0]
                    if 0 <= lead <= LEAD_TIME_MAX_WINDOW:
                        lead_time_full[e_idx] = lead
                    else:
                        late_flags[e_idx] = True
                else:
                    miss_flags[e_idx] = True

        avg_lead_second = float(np.nanmean(lead_time_full)) if not np.all(np.isnan(lead_time_full)) else 0.0
        lead_time_median = float(np.nanmedian(lead_time_full)) if not np.all(np.isnan(lead_time_full)) else 0.0
        lead_time_miss_rate = (
            float(np.sum(miss_flags)) / real_congested_edge_count if real_congested_edge_count > 0 else 0.0
        )
        lead_time_late_rate = (
            float(np.sum(late_flags)) / real_congested_edge_count if real_congested_edge_count > 0 else 0.0
        )

        pre_intervention_coverage = (
            pre_intervention_count / real_congested_edge_count
            if real_congested_edge_count > 0 else 0.0
        )


        lead_offset = int(avg_lead_second) if pre_intervention_count > 0 else 30
        lead_offset = max(10, min(lead_offset, len(pred_risks) // 4))


        raw_mae = np.mean(np.abs(pred_risks - real_occs))
        raw_rmse = np.sqrt(np.mean((pred_risks - real_occs) ** 2))

        pred_flat = pred_risks.flatten()
        real_flat = real_smoothed.flatten()
        k = int(len(pred_flat) * self.top_ratio)
        pred_top_idx = np.argsort(pred_flat)[-k:]
        real_top_idx = np.argsort(real_flat)[-k:]
        top_hit_ratio = len(np.intersect1d(pred_top_idx, real_top_idx)) / k

        pred_aligned = pred_risks[:-lead_offset, :].flatten()
        real_aligned = real_smoothed[lead_offset:, :].flatten()
        k_aligned = int(len(pred_aligned) * self.top_ratio)
        pred_top_aligned = np.argsort(pred_aligned)[-k_aligned:]
        real_top_aligned = np.argsort(real_aligned)[-k_aligned:]
        top_hit_ratio_aligned = len(np.intersect1d(pred_top_aligned, real_top_aligned)) / k_aligned


        business_mask = pred_risks >= self.business_threshold
        business_count = np.sum(business_mask)
        business_trigger_ratio = business_count / pred_risks.size


        precision_horizon = RISE_RESPONSE_MAX_WINDOW
        triggered_edge_count = 0
        precision_hit_count = 0
        for edge_idx in range(self.num_edges):
            trig_idx = np.where(business_mask[:, edge_idx])[0]
            if len(trig_idx) == 0:
                continue
            triggered_edge_count += 1
            first_trigger = trig_idx[0]
            window_end = min(first_trigger + precision_horizon, len(real_smoothed) - 1)
            future_real = real_smoothed[first_trigger:window_end + 1, edge_idx]
            if np.any(future_real >= self.real_congestion_threshold):
                precision_hit_count += 1
        precision = precision_hit_count / triggered_edge_count if triggered_edge_count > 0 else 0.0

        congested_edge_count = 0
        recall_hit_count = 0
        for edge_idx in range(self.num_edges):
            real_idx = np.where(real_cong_mask_thresh[:, edge_idx])[0]
            if len(real_idx) == 0:
                continue
            congested_edge_count += 1
            first_real = real_idx[0]
            trig_idx = np.where(business_mask[:, edge_idx])[0]
            prior_trig = trig_idx[trig_idx <= first_real]
            if len(prior_trig) > 0:
                recall_hit_count += 1
        recall = recall_hit_count / congested_edge_count if congested_edge_count > 0 else 0.0


        trigger_edge_count = 0
        success_count = 0
        edge_max_real = np.max(real_smoothed, axis=0)

        for edge_idx in range(self.num_edges):
            edge_pred_mask = business_mask[:, edge_idx]
            if not np.any(edge_pred_mask):
                continue
            trigger_edge_count += 1

            if edge_max_real[edge_idx] < self.real_congestion_threshold:
                success_count += 1

        diversion_success_rate = precision


        clear_edges_mask = np.all(real_smoothed < self.real_congestion_threshold, axis=0)
        clear_edge_indices = np.where(clear_edges_mask)[0]
        false_trigger_count = 0
        for idx in clear_edge_indices:
            if np.any(business_mask[:, idx]):
                false_trigger_count += 1
        precautionary_trigger_ratio = (
            false_trigger_count / len(clear_edge_indices)
            if len(clear_edge_indices) > 0 else 0.0
        )


        no_trigger_edges = np.where(~np.any(business_mask, axis=0))[0]
        clear_count = 0
        for idx in no_trigger_edges:
            if clear_edges_mask[idx]:
                clear_count += 1
        no_miss_rate = clear_count / len(no_trigger_edges) if len(no_trigger_edges) > 0 else 1.0


        real_std = np.std(np.mean(real_smoothed, axis=0))


        real_cong_mask = real_smoothed >= self.real_congestion_threshold
        total_congestion_time = float(np.sum(real_cong_mask))


        if np.sum(real_cong_mask) > 0:
            avg_congestion_depth = float(
                np.mean(real_smoothed[real_cong_mask] - self.real_congestion_threshold)
            )
        else:
            avg_congestion_depth = 0.0


        real_cong_durations = []
        for edge_idx in range(self.num_edges):
            edge_mask = real_cong_mask[:, edge_idx]
            diff = np.diff(np.concatenate([[0], edge_mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                real_cong_durations.append(e - s)
        real_avg_congestion_duration = np.mean(real_cong_durations) if len(real_cong_durations) > 0 else 0.0


        pred_cong_durations = []
        edge_business_mask_all = pred_risks >= self.business_threshold
        for edge_idx in range(self.num_edges):
            edge_mask = edge_business_mask_all[:, edge_idx]
            diff = np.diff(np.concatenate([[0], edge_mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                pred_cong_durations.append(e - s)
        avg_pred_duration = np.mean(pred_cong_durations) if len(pred_cong_durations) > 0 else 0.0


        recovery_times = []
        for edge_idx in range(self.num_edges):
            edge_real = real_smoothed[:, edge_idx]
            peak_idx = np.argmax(edge_real)
            peak_val = edge_real[peak_idx]
            if peak_val < self.real_congestion_threshold:
                continue
            follow_real = edge_real[peak_idx:]
            below_pos = np.where(follow_real < self.real_congestion_threshold)[0]
            if len(below_pos) > 0:
                recovery_times.append(below_pos[0])
        avg_recovery_time = np.mean(recovery_times) if len(recovery_times) > 0 else 0.0


        rise_response_times = []
        fall_recovery_times = []
        rise_response_excluded_count = 0
        fall_recovery_excluded_count = 0

        real_cong_thresh = np.quantile(real_smoothed, 1 - self.top_ratio)
        pred_cong_thresh = np.quantile(pred_risks, 1 - self.top_ratio)

        real_cong_mask = real_smoothed >= real_cong_thresh
        pred_cong_mask = pred_risks >= pred_cong_thresh

        for edge_idx in range(self.num_edges):
            real_series = real_cong_mask[:, edge_idx]
            pred_series = pred_cong_mask[:, edge_idx]

            diff = np.diff(np.concatenate([[0], real_series.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for s, e in zip(starts, ends):

                pred_before_peak = pred_series[:s + 1]
                pred_first_cong = np.where(pred_before_peak)[0]
                if len(pred_first_cong) > 0:
                    resp = s - pred_first_cong[-1]
                    if 0 <= resp <= RISE_RESPONSE_MAX_WINDOW:
                        rise_response_times.append(resp)
                    else:
                        rise_response_excluded_count += 1
                else:
                    rise_response_excluded_count += 1


                pred_after_end = pred_series[e:]
                pred_below = np.where(~pred_after_end)[0]
                if len(pred_below) > 0:
                    rec = pred_below[0]
                    if 0 <= rec <= RISE_RESPONSE_MAX_WINDOW:
                        fall_recovery_times.append(rec)
                    else:
                        fall_recovery_excluded_count += 1
                else:
                    fall_recovery_excluded_count += 1

        avg_rise_response = np.mean(rise_response_times) if len(rise_response_times) > 0 else 0.0
        avg_fall_recovery = np.mean(fall_recovery_times) if len(fall_recovery_times) > 0 else 0.0


        real_peak_max = real_occs.max()
        pred_peak_max = pred_risks.max()

        peak_pred_deviation = abs(real_peak_max - pred_peak_max) / real_peak_max if real_peak_max > 0 else 0.0


        diversion_cost_efficiency = (
            avg_congestion_depth / (business_trigger_ratio * 100)
            if business_trigger_ratio > 1e-6 else float('inf')
        )

        return {

            "raw_mae": raw_mae,
            "raw_rmse": raw_rmse,
            "top_hit_ratio": top_hit_ratio,
            "top_hit_ratio_aligned": top_hit_ratio_aligned,
            "peak_pred_deviation": peak_pred_deviation,

            "peak_attenuation_rate": peak_pred_deviation,
            "pred_quant_thresh": pred_quant_thresh,
            "pred_max": pred_risks.max(),


            "real_occupancy_std": real_std,
            "real_peak_max": real_peak_max,
            "total_congestion_time": total_congestion_time,
            "avg_congestion_depth": avg_congestion_depth,
            "real_avg_congestion_duration": real_avg_congestion_duration,
            "avg_recovery_time": avg_recovery_time,

            "avg_congestion_duration": avg_pred_duration,
            "congested_ratio": float(np.mean(real_cong_mask)),
            "congestion_event_count": len(real_cong_durations),


            "avg_lead_second": avg_lead_second,
            "lead_time_median": lead_time_median,
            "lead_time_miss_rate": lead_time_miss_rate,
            "lead_time_late_rate": lead_time_late_rate,
            "lead_time_distribution": lead_time_full,
            "pre_intervention_coverage": pre_intervention_coverage,
            "diversion_success_rate": diversion_success_rate,
            "precision": precision,
            "recall": recall,
            # 原始计数（触发边数/命中数），供跨seed做pooled precision/recall统计时使用，
            # 避免"先对每个seed各自算比例再平均"在触发边数很少时产生小样本假象
            "triggered_edge_count": int(triggered_edge_count),
            "precision_hit_count": int(precision_hit_count),
            "congested_edge_count": int(congested_edge_count),
            "recall_hit_count": int(recall_hit_count),
            "precautionary_trigger_ratio": precautionary_trigger_ratio,
            "no_miss_rate": no_miss_rate,
            "business_trigger_ratio": business_trigger_ratio,
            "total_trigger_events": int(np.sum(np.any(business_mask, axis=1))),
            "diversion_cost_efficiency": diversion_cost_efficiency,


            "avg_rise_response": avg_rise_response,
            "avg_fall_recovery": avg_fall_recovery,
            "rise_response_excluded_count": rise_response_excluded_count,
            "fall_recovery_excluded_count": fall_recovery_excluded_count,
        }

    def calculate_lead_time(self, target_edge_idx):
        if self._pred_quantile_threshold is None:
            self._compute_pred_based_threshold()

        real_thresh = self.real_congestion_threshold
        pred_thresh = self.business_threshold[target_edge_idx]

        pred_array = np.array(self.pred_risk_history)[:, target_edge_idx]
        real_array = self._sliding_mean(np.array(self.real_occupancy_history))[:, target_edge_idx]

        pred_times = np.where(pred_array >= pred_thresh)[0]
        real_times = np.where(real_array >= real_thresh)[0]

        if len(pred_times) == 0 or len(real_times) == 0:
            return np.nan
        lead = real_times[0] - pred_times[0]
        return lead if 0 <= lead <= LEAD_TIME_MAX_WINDOW else np.nan




try:
    import libsumo as traci
except ImportError:
    import traci
import sumolib
import pickle
import pandas
import xml.etree.ElementTree as ET


def stream_load_node_bbox_and_scope(node_file, network_scale_ratio):
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    node_coords = {}
    node_types = {}
    for _, elem in ET.iterparse(node_file, events=("end",)):
        if elem.tag == "node":
            nid = elem.get("id")
            x = float(elem.get("x"))
            y = float(elem.get("y"))
            node_coords[nid] = (x, y)
            node_types[nid] = elem.get("type")
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
            elem.clear()
    if network_scale_ratio >= 1.0:
        node_in_scope = set(node_coords.keys())
    else:
        x_cut = x_min + (x_max - x_min) * np.sqrt(network_scale_ratio)
        y_cut = y_min + (y_max - y_min) * np.sqrt(network_scale_ratio)
        node_in_scope = {nid for nid, (x, y) in node_coords.items() if x <= x_cut and y <= y_cut}
    return node_in_scope, node_coords, node_types


def stream_load_cropped_edges(edge_file, node_in_scope):
    edge_ids = []
    edge_from_node = {}
    edge_to_node = {}
    edge_num_lanes = {}
    free_flow_speeds = []
    edge_lengths = []
    required_node_ids = set()
    for _, elem in ET.iterparse(edge_file, events=("end",)):
        if elem.tag == "edge":
            eid = elem.get("id")
            from_n = elem.get("from")
            to_n = elem.get("to")
            if from_n in node_in_scope:
                edge_ids.append(eid)
                edge_from_node[eid] = int(from_n)
                edge_to_node[eid] = int(to_n)
                edge_num_lanes[eid] = elem.get("numLanes", "1")
                free_flow_speeds.append(float(elem.get("speed")))
                edge_lengths.append(float(elem.get("length")))
                required_node_ids.add(from_n)
                required_node_ids.add(to_n)
            elem.clear()
    return edge_ids, edge_from_node, edge_to_node, edge_num_lanes, free_flow_speeds, edge_lengths, required_node_ids


def build_cropped_sumo_network(netconvert_exe, output_net_file, node_coords, node_types,
                                edge_ids, edge_from_node, edge_to_node, edge_num_lanes,
                                free_flow_speeds, edge_lengths, required_node_ids):
    import subprocess
    tmp_nod = output_net_file + ".tmp.nod.xml"
    tmp_edg = output_net_file + ".tmp.edg.xml"

    with open(tmp_nod, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')
        for nid in sorted(required_node_ids, key=lambda v: int(v)):
            x, y = node_coords[nid]
            ntype = node_types.get(nid)
            if ntype:
                f.write(f'    <node id="{nid}" x="{x}" y="{y}" type="{ntype}"/>\n')
            else:
                f.write(f'    <node id="{nid}" x="{x}" y="{y}"/>\n')
        f.write('</nodes>\n')

    with open(tmp_edg, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')
        for i, eid in enumerate(edge_ids):
            f.write(
                f'    <edge id="{eid}" from="{edge_from_node[eid]}" to="{edge_to_node[eid]}" '
                f'numLanes="{edge_num_lanes[eid]}" speed="{free_flow_speeds[i]}" length="{edge_lengths[i]}"/>\n'
            )
        f.write('</edges>\n')

    subprocess.run(
        [netconvert_exe, "-n", tmp_nod, "-e", tmp_edg, "-o", output_net_file,
         "--no-warnings", "--tls.guess", "true"],
        check=True, capture_output=True
    )
    os.remove(tmp_nod)
    os.remove(tmp_edg)


def build_edge_connectivity_from_net(net_file):
    edge_out = defaultdict(set)
    for _, elem in ET.iterparse(net_file, events=("end",)):
        if elem.tag == "connection":
            frm = elem.get("from")
            to = elem.get("to")
            if frm and to and not frm.startswith(":") and not to.startswith(":"):
                edge_out[frm].add(to)
        elem.clear()
    return edge_out


def build_edge_reachable_nodes(edge_ids, edge_out_connections, edge_to_node):
    reachable = {}
    for start in edge_ids:
        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in edge_out_connections.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        reachable[start] = frozenset(edge_to_node[e] for e in visited if e in edge_to_node)
    return reachable


def resolve_from_edge(o_node, from_edge_match, edge_id_to_idx, junction_out_edges):
    if from_edge_match:
        cand = from_edge_match.group(1)
        if cand in edge_id_to_idx:
            return cand
    for eid in junction_out_edges.get(o_node, ()):
        if eid in edge_id_to_idx:
            return eid
    return None


def count_eligible_trips(trip_file, valid_node_ids, edge_reachable_nodes, edge_id_to_idx, junction_out_edges, max_depart):
    depart_pattern = re.compile(r'depart="([^"]+)"')
    junction_pattern = re.compile(r'fromJunction="([^"]+)"')
    to_junction_pattern = re.compile(r'toJunction="([^"]+)"')
    from_edge_pattern = re.compile(r'from="([^"]+)"')
    count = 0
    with open(trip_file, "r", encoding="utf-8") as f:
        for line in f:
            if '<trip ' not in line:
                continue
            depart_match = depart_pattern.search(line)
            if not depart_match or float(depart_match.group(1)) > max_depart:
                break
            junc_match = junction_pattern.search(line)
            to_junc_match = to_junction_pattern.search(line)
            if not junc_match or junc_match.group(1) not in valid_node_ids:
                continue
            if not to_junc_match:
                continue
            o_node = int(junc_match.group(1))
            d_node = int(to_junc_match.group(1))
            from_edge_match = from_edge_pattern.search(line)
            from_eid = resolve_from_edge(o_node, from_edge_match, edge_id_to_idx, junction_out_edges)
            if from_eid is None:
                continue
            if d_node in edge_reachable_nodes.get(from_eid, ()):
                count += 1
    return count


MAX_DEPARTURES_PER_SECOND = 2000


def stream_sample_trips(trip_file, filtered_trips_path, valid_node_ids, edge_reachable_nodes, edge_from_node_lookup,
                         edge_id_to_idx, max_depart, sample_prob, seed):
    import random
    rng = random.Random(seed)
    id_pattern = re.compile(r'id="([^"]+)"')
    depart_pattern = re.compile(r'depart="([^"]+)"')
    junction_pattern = re.compile(r'fromJunction="([^"]+)"')
    to_junction_pattern = re.compile(r'toJunction="([^"]+)"')
    from_edge_pattern = re.compile(r'from="([^"]+)"')

    junction_out_edges = defaultdict(list)
    for eid, n in edge_from_node_lookup.items():
        junction_out_edges[n].append(eid)

    kept_trips = []
    with open(trip_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if '<trip ' not in line:
                continue
            depart_match = depart_pattern.search(line)
            if not depart_match or float(depart_match.group(1)) > max_depart:
                break
            junc_match = junction_pattern.search(line)
            to_junc_match = to_junction_pattern.search(line)
            if not junc_match or junc_match.group(1) not in valid_node_ids:
                continue
            if not to_junc_match:
                continue
            if rng.random() >= sample_prob:
                continue

            id_match = id_pattern.search(line)
            from_edge_match = from_edge_pattern.search(line)
            if not id_match:
                continue

            o_node = int(junc_match.group(1))
            d_node = int(to_junc_match.group(1))
            from_eid = resolve_from_edge(o_node, from_edge_match, edge_id_to_idx, junction_out_edges)
            if from_eid is None:
                continue
            if d_node not in edge_reachable_nodes.get(from_eid, ()):
                continue

            kept_trips.append((float(depart_match.group(1)), id_match.group(1), o_node, d_node, from_eid))

    bucket_count = {}
    pointer = 0
    smoothed_departs = []
    for depart, vid, o_node, d_node, from_eid in kept_trips:
        orig_second = int(depart)
        target = max(orig_second, pointer)
        while bucket_count.get(target, 0) >= MAX_DEPARTURES_PER_SECOND:
            target += 1
        bucket_count[target] = bucket_count.get(target, 0) + 1
        pointer = target
        frac = depart - orig_second if target == orig_second else bucket_count[target] / (MAX_DEPARTURES_PER_SECOND + 1.0)
        smoothed_departs.append(target + frac)

    veh_ids = []
    veh_rows = []
    with open(filtered_trips_path, "w", encoding="utf-8") as fout:
        fout.write('<?xml version=\'1.0\' encoding=\'utf-8\'?>\n')
        fout.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n')
        for (depart, vid, o_node, d_node, from_eid), new_depart in zip(kept_trips, smoothed_departs):
            fout.write(f'    <trip id="{vid}" depart="{new_depart:.2f}" fromJunction="{o_node}" toJunction="{d_node}" from="{from_eid}" />\n')
            veh_ids.append(vid)
            veh_rows.append((o_node, d_node, edge_id_to_idx[from_eid]))
        fout.write('    </routes>\n')
    return veh_ids, veh_rows


def parse_filtered_trips(filtered_trips_path, edge_id_to_idx, valid_node_ids, edge_reachable_nodes):
    id_pattern = re.compile(r'id="([^"]+)"')
    junction_pattern = re.compile(r'fromJunction="([^"]+)"')
    to_junction_pattern = re.compile(r'toJunction="([^"]+)"')
    from_edge_pattern = re.compile(r'from="([^"]+)"')
    veh_ids = []
    veh_rows = []
    skipped = 0
    with open(filtered_trips_path, "r", encoding="utf-8") as f:
        for line in f:
            if '<trip ' not in line:
                continue
            id_match = id_pattern.search(line)
            junc_match = junction_pattern.search(line)
            to_junc_match = to_junction_pattern.search(line)
            from_edge_match = from_edge_pattern.search(line)
            if not (id_match and junc_match and to_junc_match and from_edge_match):
                continue
            from_eid = from_edge_match.group(1)
            if from_eid not in edge_id_to_idx:
                skipped += 1
                continue
            if junc_match.group(1) not in valid_node_ids:
                skipped += 1
                continue
            if int(to_junc_match.group(1)) not in edge_reachable_nodes.get(from_eid, ()):
                skipped += 1
                continue
            veh_ids.append(id_match.group(1))
            veh_rows.append((int(junc_match.group(1)), int(to_junc_match.group(1)), edge_id_to_idx[from_eid]))
    if skipped:
        print(f"⚠️ 已跳过 {skipped} 条与当前裁剪路网不匹配的缓存行程（可能来自旧版缓存文件）")
    return veh_ids, veh_rows


def build_adjacency_matrix(edge_ids, edge_from_node, edge_to_node, edge_id_to_idx):
    num_edges = len(edge_ids)
    from_node_to_indices = {}
    for j, eid in enumerate(edge_ids):
        from_node_to_indices.setdefault(edge_from_node[eid], []).append(j)
    rows = []
    cols = []
    for i, eid in enumerate(edge_ids):
        dst_node = edge_to_node[eid]
        for j in from_node_to_indices.get(dst_node, []):
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(num_edges, num_edges))
    return adj


VEH_PRE_MAP_DTYPE = np.dtype([('o_node', np.int32), ('d_node', np.int32), ('start_idx', np.int32)])


class VehPreMap:
    __slots__ = ('_index', '_data')

    def __init__(self, index, data):
        self._index = index
        self._data = data

    def get(self, vid, default=None):
        row = self._index.get(vid)
        if row is None:
            return default
        rec = self._data[row]
        return (int(rec['o_node']), int(rec['d_node']), int(rec['start_idx']))

    def __len__(self):
        return len(self._data)


class MarginalCostRoutingAllocator(EndogenousRouteAllocator):
    def __init__(self, *args, mc_iterations=5, bpr_alpha=0.15, bpr_power=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_iterations = mc_iterations
        self.bpr_alpha = bpr_alpha
        self.bpr_power = bpr_power

    def allocate_vehicles(self, vehicle_list, free_flow_speeds, edge_lengths):
        if len(vehicle_list) == 0:
            return {}
        self.congestion_tracker.step_forward()
        self._step_counter += 1

        od_groups = defaultdict(list)
        for veh in vehicle_list:
            veh_id, from_e, to_e, _ = veh
            od_groups[(from_e, to_e)].append((veh_id, from_e, to_e))

        allocation_results = {}
        all_batch_edges = []
        all_batch_offsets = []

        for od_key, vehs in od_groups.items():
            if od_key not in self.od_candidate_paths:
                for vid, fe, te in vehs:
                    allocation_results[vid] = None
                continue

            paths = self._path_edges_cache[od_key]
            num_paths = len(paths)
            n_vehs = len(vehs)
            if num_paths == 0:
                for vid, fe, te in vehs:
                    allocation_results[vid] = None
                continue

            flow_split = np.full(num_paths, n_vehs / num_paths, dtype=np.float64)

            for _ in range(self.mc_iterations):
                marg_costs = np.zeros(num_paths, dtype=np.float64)
                for p_idx in range(num_paths):
                    path_edges = paths[p_idx]
                    t0 = float(np.sum(edge_lengths[path_edges] / np.maximum(free_flow_speeds[path_edges], 1e-3)))
                    capacity = max(float(np.mean(self.congestion_tracker.edge_capacity[path_edges])), 1e-3)
                    ratio = flow_split[p_idx] / capacity
                    travel_time = t0 * (1 + self.bpr_alpha * ratio ** self.bpr_power)
                    dtdflow = t0 * self.bpr_alpha * self.bpr_power * (ratio ** max(self.bpr_power - 1, 0)) / capacity
                    marg_costs[p_idx] = travel_time + flow_split[p_idx] * dtdflow

                if num_paths > 1:
                    mean_cost = np.mean(marg_costs)
                    adjustment = mean_cost - marg_costs
                    step = 0.1 * flow_split * np.sign(adjustment)
                    flow_split = np.maximum(flow_split + step, 0.0)
                    total = np.sum(flow_split)
                    if total > 1e-6:
                        flow_split = flow_split / total * n_vehs

            alloc_cumsum = np.cumsum(flow_split)
            veh_indices = np.arange(n_vehs, dtype=np.float64) + 0.5
            path_idx_arr = np.searchsorted(alloc_cumsum, veh_indices)
            path_idx_arr = np.clip(path_idx_arr, 0, num_paths - 1)

            for idx, (vid, fe, te) in enumerate(vehs):
                allocation_results[vid] = int(path_idx_arr[idx])

            path_edges_cache = self._path_edges_cache[od_key]
            path_offsets_cache = self._path_offsets_cache[od_key]
            for p_idx in path_idx_arr:
                all_batch_edges.append(path_edges_cache[p_idx])
                all_batch_offsets.append(path_offsets_cache[p_idx])

        if all_batch_edges:
            all_edges = np.concatenate(all_batch_edges)
            all_offsets = np.concatenate(all_batch_offsets)
            self.congestion_tracker.update_allocation(all_edges, all_offsets)

        self.total_demand += len(vehicle_list)
        self.total_count += 1
        return allocation_results


class MPCRoutingAllocator(EndogenousRouteAllocator):
    def __init__(self, *args, control_interval=30, candidate_ratios=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_interval = control_interval
        self.candidate_ratios = candidate_ratios if candidate_ratios is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        self._current_ratio = 0.0
        self._mpc_step_counter = 0

    def _path_avg_risk(self, od_key, base_risk):
        flat_edges, _ = self._path_congest_cache[od_key]
        reduce_idx = self._path_reduce_idx_cache[od_key]
        counts = np.add.reduceat(np.ones_like(flat_edges, dtype=np.float64), reduce_idx)
        return np.add.reduceat(base_risk[flat_edges], reduce_idx) / counts

    def _evaluate_ratio_cost(self, ratio, od_key, n_vehs, path_avg_risk, primary_idx, alt_idx):
        diverted = ratio * n_vehs
        kept = n_vehs - diverted
        return kept * path_avg_risk[primary_idx] + diverted * path_avg_risk[alt_idx]

    def allocate_vehicles(self, vehicle_list, free_flow_speeds, edge_lengths):
        if len(vehicle_list) == 0:
            return {}
        self.congestion_tracker.step_forward()
        self._step_counter += 1
        self._mpc_step_counter += 1

        od_groups = defaultdict(list)
        for veh in vehicle_list:
            veh_id, from_e, to_e, _ = veh
            od_groups[(from_e, to_e)].append((veh_id, from_e, to_e))

        allocation_results = {}
        all_batch_edges = []
        all_batch_offsets = []

        recompute = (self._mpc_step_counter % self.control_interval == 0)
        base_risk = self.congestion_tracker.calculate_congestion_risk()

        for od_key, vehs in od_groups.items():
            if od_key not in self.od_candidate_paths:
                for vid, fe, te in vehs:
                    allocation_results[vid] = None
                continue

            paths = self._path_edges_cache[od_key]
            num_paths = len(paths)
            n_vehs = len(vehs)
            if num_paths == 0:
                for vid, fe, te in vehs:
                    allocation_results[vid] = None
                continue

            path_avg_risk = self._path_avg_risk(od_key, base_risk)
            primary_idx = int(np.argmin(path_avg_risk))
            alt_idx = int(np.argmax(path_avg_risk)) if num_paths > 1 else primary_idx

            if recompute or self._current_ratio is None:
                costs = [
                    self._evaluate_ratio_cost(r, od_key, n_vehs, path_avg_risk, primary_idx, alt_idx)
                    for r in self.candidate_ratios
                ]
                best_idx = int(np.argmin(costs))
                self._current_ratio = self.candidate_ratios[best_idx]

            path_idx_arr = np.full(n_vehs, primary_idx, dtype=np.int64)
            n_divert = int(round(self._current_ratio * n_vehs))
            if n_divert > 0 and alt_idx != primary_idx:
                path_idx_arr[-n_divert:] = alt_idx

            for idx, (vid, fe, te) in enumerate(vehs):
                allocation_results[vid] = int(path_idx_arr[idx])

            path_edges_cache = self._path_edges_cache[od_key]
            path_offsets_cache = self._path_offsets_cache[od_key]
            for p_idx in path_idx_arr:
                all_batch_edges.append(path_edges_cache[p_idx])
                all_batch_offsets.append(path_offsets_cache[p_idx])

        if all_batch_edges:
            all_edges = np.concatenate(all_batch_edges)
            all_offsets = np.concatenate(all_batch_offsets)
            self.congestion_tracker.update_allocation(all_edges, all_offsets)

        self.total_demand += len(vehicle_list)
        self.total_count += 1
        return allocation_results


from tqdm import tqdm
import time
def run_single_simulation(strategy_name, allocator, net, edges, edge_id_to_idx, edge_idx_to_id,
                          free_flow_speeds, edge_lengths, edge_from_node, veh_map_local,
                          trips_file, sim_duration, seed, output_tripinfo, net_file=None):

    threshold = allocator.congestion_threshold

    evaluator = CongestionPredictionEvaluator(
        num_edges=len(edges),
        business_threshold=allocator.congestion_threshold,
        top_ratio=0.1
    )
    SUMO_EXE = r"D:\安装程序\sumo-win64-1.27.1\sumo-1.27.1\bin\sumo.exe"

    if net_file is None:
        try:
            net_file = NET_FILE
        except NameError:
            import os
            for fname in os.listdir('.'):
                if fname.endswith('_cropped_0.2.net.xml'):
                    net_file = fname
                    break
            else:
                net_file = 'network.net.xml'

    sumo_cmd = [
        SUMO_EXE,
        "-n", net_file,
        "-r", trips_file,
        "--begin", "0",
        "--end", str(sim_duration),
        "--step-length", "1",
        "--junction-taz",
        "--no-step-log",
        "--xml-validation", "never",
        "--no-warnings",
        "--time-to-teleport", "300",
        "--seed", str(seed),
        "--tripinfo-output", output_tripinfo
    ]

    traci.start(sumo_cmd)
    print(f"⏳ SUMO已启动，跳过边订阅以加速...")
    sim_step = 0
    assigned_count = 0
    total_algo_time = 0.0
    path_edge_ids = allocator._path_edge_ids_cache
    path_edge_pos = allocator._path_edge_pos_cache

    departed_vehicle_ids = set()
    halting_sum = 0.0
    halting_max = 0
    teleport_total = 0
    saturation_fraction_sum = 0.0
    saturation_fraction_max = 0.0
    diag_step_count = 0
    last_real_occupancy = np.zeros(len(edges), dtype=np.float32)
    last_halting_count = 0
    max_total_steps = sim_duration + CLEARANCE_DURATION

    print(f"⏳ 开始仿真，总步数: {max_total_steps}")
    with tqdm(total=max_total_steps, desc=f"📊 {strategy_name}", unit="步", ncols=100) as pbar:
        while sim_step < max_total_steps:
            traci.simulationStep()
            sim_step += 1
            pbar.update(1)

            if sim_step >= sim_duration and traci.simulation.getMinExpectedNumber() == 0:
                break

            if sim_step % 10 == 0:
                real_occupancy = np.zeros(len(edges), dtype=np.float32)
                halting_count = 0
                for i, e in enumerate(edges):
                    try:
                        occ = traci.edge.getLastStepOccupancy(e) / 100.0
                        real_occupancy[i] = min(occ, 1.0)
                        halting_count += traci.edge.getLastStepHaltingNumber(e)
                    except:
                        pass
                last_real_occupancy = real_occupancy
                last_halting_count = halting_count
            real_occupancy = last_real_occupancy
            halting_count = last_halting_count

            try:
                teleport_count = traci.simulation.getStartingTeleportNumber()
            except:
                teleport_count = 0
            saturation_fraction = float(np.mean(real_occupancy >= 0.95))
            halting_sum += halting_count
            halting_max = max(halting_max, halting_count)
            teleport_total += teleport_count
            saturation_fraction_sum += saturation_fraction
            saturation_fraction_max = max(saturation_fraction_max, saturation_fraction)
            diag_step_count += 1



            t0 = time.perf_counter()
            if hasattr(allocator, 'update_traffic_data') and hasattr(allocator, 'step_advance'):
                allocator.update_traffic_data(real_occupancy)
                allocator.step_advance()
            total_algo_time += time.perf_counter() - t0



            departed = traci.simulation.getDepartedIDList()
            departed_vehicle_ids.update(departed)

            if sim_step <= 5 and len(departed) > 0:
                print(f"  步{sim_step}: {len(departed)}辆车出发")

            t0 = time.perf_counter()

            if sim_step > WARMUP_DURATION:
                if isinstance(allocator, EndogenousRouteAllocator):

                    pred_risk = allocator.calculate_congestion_risk()
                    evaluator.record_step(pred_risk, real_occupancy)
                elif hasattr(allocator, 'calculate_congestion_risk'):

                    pred_risk = allocator.calculate_congestion_risk()
                    evaluator.record_step(pred_risk, real_occupancy)
            total_algo_time += time.perf_counter() - t0


            if len(departed) > 0:
                vehicle_list = []
                vid_od_map = {}
                for vid in departed:
                    info = veh_map_local.get(vid)
                    if info is None:
                        continue
                    o_node, d_node, start_idx = info
                    vehicle_list.append((vid, o_node, d_node, start_idx))
                    vid_od_map[vid] = (o_node, d_node)

                if len(vehicle_list) > 0:
                    t0 = time.perf_counter()
                    if isinstance(allocator, EndogenousRouteAllocator):
                        alloc_results = allocator.allocate_vehicles(vehicle_list, free_flow_speeds, edge_lengths)
                    else:

                        alloc_results = allocator.allocate_vehicles(vehicle_list)
                    total_algo_time += time.perf_counter() - t0

                    for vid, path_idx in alloc_results.items():
                        if path_idx is None:
                            continue
                        try:
                            current_e = traci.vehicle.getRoadID(vid)
                        except:
                            continue
                        od_key = vid_od_map[vid]
                        ids_list = path_edge_ids.get(od_key)
                        pos_list = path_edge_pos.get(od_key)
                        if ids_list is None:
                            continue
                        pos_map = pos_list[path_idx]
                        real_pos = pos_map.get(current_e, -1)
                        if real_pos == -1:
                            continue
                        valid_route = ids_list[path_idx][real_pos:]
                        if len(valid_route) < 2:
                            continue
                        try:
                            traci.vehicle.setRoute(vid, valid_route)
                            assigned_count += 1
                        except:
                            pass
    traci.close()
    gc.collect()
    metrics = evaluator.calculate_metrics()


    avg_real_occ = np.mean(evaluator.real_occupancy_history, axis=0)
    top_edges = np.argsort(avg_real_occ)[-5:]
    lead_times = [evaluator.calculate_lead_time(e) for e in top_edges]
    valid_lead_times = [t for t in lead_times if not np.isnan(t) and t > 0]
    lead_time = np.mean(valid_lead_times) if len(valid_lead_times) > 0 else 0


    import xml.etree.ElementTree as ET
    trip_durations = []
    trip_departures = []
    completed_vehicle_ids = set()
    tree = ET.parse(output_tripinfo)
    for trip in tree.getroot().findall('tripinfo'):
        duration = float(trip.get('duration', 0))
        trip_durations.append(duration)
        vid = trip.get('id')
        if vid is not None:
            completed_vehicle_ids.add(vid)

    avg_trip_time = np.mean(trip_durations) if len(trip_durations) > 0 else 0
    incomplete_vehicle_count = len(departed_vehicle_ids - completed_vehicle_ids)
    incomplete_ratio = (
        incomplete_vehicle_count / len(departed_vehicle_ids) if len(departed_vehicle_ids) > 0 else 0.0
    )
    network_metrics = {
        "avg_trip_time": avg_trip_time,
        "total_completed": len(trip_durations),
        "incomplete_vehicle_count": incomplete_vehicle_count,
        "incomplete_ratio": incomplete_ratio,
        "mean_halting_vehicles": halting_sum / diag_step_count if diag_step_count > 0 else 0.0,
        "max_halting_vehicles": halting_max,
        "total_spillback_teleports": teleport_total,
        "mean_saturation_fraction": saturation_fraction_sum / diag_step_count if diag_step_count > 0 else 0.0,
        "max_saturation_fraction": saturation_fraction_max,
        "traffic_loading_model": TRAFFIC_LOADING_MODEL,
        "real_occupancy_history": np.array(evaluator.real_occupancy_history),
    }


    efficiency_metrics = {
        "total_algo_time": total_algo_time,
        "avg_step_ms": total_algo_time / sim_duration * 1000,
        "throughput": sim_duration / total_algo_time if total_algo_time > 0 else 0,

        "unit_node_step_ms": (total_algo_time / sim_duration * 1000) / len(edges)
    }


    return metrics, lead_time, assigned_count, network_metrics, efficiency_metrics



import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from matplotlib import rcParams





from matplotlib import rcParams


from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans']

rcParams['axes.unicode_minus'] = False

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

rcParams['figure.dpi'] = 150

STRATEGY_NAMES = [
    "Original Endogenous",
    "D³STN ",
    "STGCN Baseline",
    "DCRNN Baseline",
    "ST-ResNet Baseline",
    "Marginal-Cost Routing",
    "MPC Routing"
]
COLORS = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#2ca02c', '#17becf']
MARKERS = ['o', 's', 'D', 'v', 'p', '^', 'X']


def plot_bar_single_threshold(thresh, metrics_list, save_path="bar_thresh.pdf"):

    metric_keys = [

        "avg_trip_time",
        "real_occupancy_std",
        "congested_ratio",
        "total_congestion_time",
        "avg_congestion_depth",
        "real_avg_congestion_duration",
        "avg_recovery_time",

        "avg_lead_second",
        "pre_intervention_coverage",
        "diversion_success_rate",
        "precautionary_trigger_ratio",
        "business_trigger_ratio",
        "avg_rise_response",
        "avg_fall_recovery",

        "top_hit_ratio",
        "peak_pred_deviation",
        "raw_mae",

        "avg_step_ms"
    ]
    metric_labels = [
        "Avg Trip Time (s)",
        "Traffic Flow Std",
        "Congestion Ratio",
        "Total Congestion Time (edge·s)",
        "Avg Congestion Depth",
        "Avg Congestion Duration (s)",
        "Peak Recovery Time (s)",
        "Avg Lead Time (s)",
        "Pre-intervention Coverage",
        "Diversion Success Rate",
        "precautionary_trigger_ratio",
        "Diversion Trigger Ratio",
        "Rise Response Time (s)",
        "Fall Recovery Time (s)",
        "Top-10% Hit Rate",
        "Peak Prediction Deviation",
        "Mean Absolute Error",
        "Avg Step Time (ms)"
    ]

    x = np.arange(len(metrics_list))
    width = 0.65
    base_name = save_path.rsplit('.', 1)[0]

    for key, label in zip(metric_keys, metric_labels):
        fig, ax = plt.subplots(figsize=(7, 5))
        values = [m.get(key, 0.0) for m in metrics_list]
        ax.bar(x, values, width, color=COLORS, edgecolor='black', linewidth=0.8)

        ax.set_title(f"{label}\n(Threshold = {thresh})", fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_NAMES, rotation=35, ha='right', fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{base_name}_{key}.pdf", format='pdf', bbox_inches='tight')
        plt.close()


def plot_trend_all_thresholds(thresholds, all_metrics, save_path="trend_all.pdf"):
    metric_keys = [

        "avg_trip_time",
        "real_occupancy_std",
        "congested_ratio",
        "total_congestion_time",
        "avg_congestion_depth",
        "real_avg_congestion_duration",
        "avg_recovery_time",

        "avg_lead_second",
        "pre_intervention_coverage",
        "diversion_success_rate",
        "precautionary_trigger_ratio",
        "business_trigger_ratio",
        "avg_rise_response",
        "avg_fall_recovery",

        "top_hit_ratio",
        "peak_pred_deviation",
        "raw_mae",

        "avg_step_ms"
    ]
    metric_labels = [
        "Avg Trip Time (s)",
        "Traffic Flow Std",
        "Congestion Ratio",
        "Total Congestion Time (edge·s)",
        "Avg Congestion Depth",
        "Avg Congestion Duration (s)",
        "Peak Recovery Time (s)",
        "Avg Lead Time (s)",
        "Pre-intervention Coverage",
        "Diversion Success Rate",
        "precautionary_trigger_ratio",
        "Diversion Trigger Ratio",
        "Rise Response Time (s)",
        "Fall Recovery Time (s)",
        "Top-10% Hit Rate",
        "Peak Prediction Deviation",
        "Mean Absolute Error",
        "Avg Step Time (ms)"
    ]

    base_name = save_path.rsplit('.', 1)[0]

    for key, label in zip(metric_keys, metric_labels):
        fig, ax = plt.subplots(figsize=(7, 5))

        n_strategies = len(all_metrics[0])
        for s_idx in range(n_strategies):
            s_name = STRATEGY_NAMES[s_idx] if s_idx < len(STRATEGY_NAMES) else f"Strategy {s_idx + 1}"
            values = [all_metrics[t_idx][s_idx].get(key, 0.0) for t_idx in range(len(thresholds))]
            color = COLORS[s_idx] if s_idx < len(COLORS) else '#333333'
            marker = MARKERS[s_idx] if s_idx < len(MARKERS) else 'o'
            ax.plot(thresholds, values, color=color, marker=marker,
                    linewidth=1.5, markersize=5, label=s_name)

        ax.set_xlabel("Congestion Threshold", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} vs Threshold", fontweight='bold', fontsize=12)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=True, loc='best', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{base_name}_{key}.pdf", format='pdf', bbox_inches='tight')
        plt.close()


def calculate_threshold_sensitivity(thresholds, trigger_ratios):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if np.allclose(thresholds, thresholds[0]):
        return 0.0
    slope, _, _, _, _ = linregress(thresholds, trigger_ratios)
    return -slope


def run_sensitivity_sweep(param_name, param_values, base_config):
    results = []
    for value in param_values:
        allocator_kwargs = dict(base_config.get("allocator_kwargs", {}))
        od_candidate_paths = base_config["od_candidate_paths"]

        if param_name == "smooth_window":
            allocator_kwargs["smooth_window"] = value
        elif param_name == "time_decay_factor":
            allocator_kwargs["time_decay_factor"] = value
        elif param_name == "congestion_threshold":
            allocator_kwargs["congestion_threshold"] = value
        elif param_name == "gamma_max":
            allocator_kwargs["gamma_max"] = value
        elif param_name == "feedback_gain":
            allocator_kwargs["feedback_gain"] = value
        elif param_name == "candidate_path_count":
            od_candidate_paths = {
                od_key: paths[:value] for od_key, paths in od_candidate_paths.items()
            }
        else:
            raise ValueError(f"unsupported sweep parameter: {param_name}")

        allocator = EndogenousRouteAllocator(
            num_edges=base_config["num_edges"],
            od_candidate_paths=od_candidate_paths,
            **allocator_kwargs
        )
        allocator.set_edge_capacity(base_config["edge_capacity"])
        allocator.precompute_path_data(
            base_config["free_flow_speeds"], base_config["edge_lengths"], base_config["edge_idx_to_id"]
        )

        try:
            metrics, lead_time, assigned_count, network_metrics, efficiency_metrics = run_single_simulation(
                f"sweep_{param_name}_{value}",
                allocator,
                base_config["net"], base_config["edges"], base_config["edge_id_to_idx"], base_config["edge_idx_to_id"],
                base_config["free_flow_speeds"], base_config["edge_lengths"], base_config["edge_from_node"],
                base_config["veh_map"], base_config["trips_file"], base_config["sim_duration"], base_config["seed"],
                f"{base_config.get('output_prefix', 'tripinfo_sweep')}_{param_name}_{value}.xml"
            )
        except Exception as sweep_run_error:
            print(f"⚠️ sweep_{param_name}_{value} 仿真崩溃，跳过该组合: {sweep_run_error}")
            try:
                traci.close()
            except Exception:
                pass
            gc.collect()
            row = {"param_name": param_name, "param_value": value, "sweep_run_failed": True}
            results.append(row)
            continue

        row = {"param_name": param_name, "param_value": value, "sweep_run_failed": False}
        row.update(metrics)
        row.update(network_metrics)
        row.update(efficiency_metrics)
        row["lead_time_top_edges"] = lead_time
        row["assigned_count"] = assigned_count
        row.pop("real_occupancy_history", None)
        row.pop("lead_time_distribution", None)
        results.append(row)

    return pandas.DataFrame(results)


def run_2axis_sensitivity_sweep(param_a_name, param_a_values, param_b_name, param_b_values, base_config):
    checkpoint_path = f"sweep_checkpoint_{param_a_name}_{param_b_name}.csv"
    requested_combos = {(value_a, value_b) for value_a in param_a_values for value_b in param_b_values}
    completed_combos = set()
    checkpoint_df = pandas.DataFrame()
    if os.path.exists(checkpoint_path):
        checkpoint_df = pandas.read_csv(checkpoint_path)
        for _, existing_row in checkpoint_df.iterrows():
            combo = (existing_row[f"{param_a_name}_value"], existing_row["param_value"])
            if combo in requested_combos and not bool(existing_row.get("sweep_run_failed", False)):
                completed_combos.add(combo)
    for value_a in param_a_values:
        for value_b in param_b_values:
            if (value_a, value_b) in completed_combos:
                continue
            nested_config = dict(base_config)
            nested_config["allocator_kwargs"] = dict(base_config.get("allocator_kwargs", {}))
            nested_config["allocator_kwargs"][param_a_name] = value_a
            df_single = run_sensitivity_sweep(param_b_name, [value_b], nested_config)
            df_single[f"{param_a_name}_value"] = value_a
            checkpoint_df = pandas.concat([checkpoint_df, df_single], ignore_index=True)
            checkpoint_df.to_csv(checkpoint_path, mode="w", header=True, index=False)
    if checkpoint_df.empty:
        return pandas.DataFrame()
    result_mask = checkpoint_df.apply(
        lambda r: (r[f"{param_a_name}_value"], r["param_value"]) in requested_combos, axis=1
    )
    return checkpoint_df[result_mask].reset_index(drop=True)


if __name__ == "__main__":
    import pickle
    import xml.etree.ElementTree as ET
    import pandas
    import numpy as np
    from collections import defaultdict
    from numba import jit
    import torch
    import torch.nn as nn
    import time
    from tqdm import tqdm
    try:
        import libsumo as traci
    except ImportError:
        import traci
    import sumolib




    REAL_OCC_MEAN = 0.0035
    REAL_OCC_STD = 0.0015
    REAL_OCC_MIN = 0.0010
    REAL_OCC_MAX = 0.0065




    class STGCN_BaselineAllocator:
        def __init__(self, num_edges, od_candidate_paths, adjacency_matrix, congestion_threshold=0.4, seq_len=12,
                     pred_steps=1,
                     update_interval=1, scaler_mean=None, scaler_std=None, pretrained_path=None,
                     train_interval=20, train_batch_size=8, replay_capacity=600, learning_rate=1e-3):
            self.num_edges = num_edges
            self.congestion_threshold = congestion_threshold
            self.seq_len = seq_len
            self.pred_steps = pred_steps
            self.update_interval = update_interval
            self.step_counter = 0

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            K = 1
            scaled_adj_coo, lap_n = calculate_normalized_laplacian(adjacency_matrix)
            cheb_polys = chebyshev_polynomials(scaled_adj_coo, lap_n, K, self.device)
            del scaled_adj_coo


            self.model = STGCN(
                num_nodes=num_edges,
                in_channels=1,
                spatial_channels=8,
                out_channels=32,
                kernel_size=3,
                seq_len=seq_len,
                pred_steps=pred_steps,
                graph_conv_type='cheb',
                graph_conv_params=(K, cheb_polys),
                scaler_mean=scaler_mean,
                scaler_std=scaler_std
            ).to(self.device)

            if pretrained_path is not None:
                self.model.load_model(pretrained_path, device=self.device)
            self.model.eval()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
            self.train_interval = train_interval
            self.train_batch_size = train_batch_size
            self.replay_capacity = replay_capacity
            self._replay_buffer = []

            self.history_traffic = np.zeros((seq_len, num_edges), dtype=np.float32)
            init_val = scaler_mean if scaler_mean is not None else congestion_threshold
            self._cached_risk = np.full(num_edges, init_val, dtype=np.float32)

            self._path_priority_cache = {}
            self._path_edge_ids_cache = {}
            self._path_edge_pos_cache = {}
            self._path_edges_cache = {}
            self._path_offsets_cache = {}
            self._path_congest_cache = {}
            self._path_reduce_idx_cache = {}

        def set_edge_capacity(self, capacity_array):
            self.capacity = capacity_array

        def precompute_path_data(self, *args, **kwargs):
            pass

        def step_advance(self):
            self.step_counter += 1

        def calculate_congestion_risk(self):
            if self.step_counter % self.train_interval == 0:
                self._online_train()
            if self.step_counter % self.update_interval == 0:
                self._predict_congestion()
            return self._cached_risk

        def _online_train(self):
            needed = self.seq_len + self.pred_steps
            n = len(self._replay_buffer)
            if n < needed + self.train_batch_size:
                return

            starts = np.random.randint(0, n - needed + 1, size=self.train_batch_size)
            x_batch, y_batch = [], []
            for s in starts:
                window = self._replay_buffer[s:s + needed]
                x_win = np.stack(window[:self.seq_len], axis=0).T
                y_win = np.stack(window[self.seq_len:needed], axis=0).T
                x_batch.append(x_win)
                y_batch.append(y_win)

            x_tensor = torch.from_numpy(np.stack(x_batch)[:, None, :, :]).float().to(self.device)
            y_tensor = torch.from_numpy(np.stack(y_batch)).float().to(self.device)

            self.model.train()
            with torch.enable_grad():
                self.optimizer.zero_grad()
                pred = self.model(x_tensor)
                loss = self.criterion(pred, y_tensor)
                loss.backward()
                self.optimizer.step()
            self.model.eval()

        def _predict_congestion(self):
            with torch.no_grad():

                x = torch.from_numpy(self.history_traffic.T[None, None, ...]).float().to(self.device)
                if self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred = self.model(x)
                else:
                    pred = self.model(x)

                self._cached_risk = pred.cpu().numpy().squeeze()
                self._cached_risk = np.clip(self._cached_risk, 0.0, REAL_OCC_MAX * 1.5)

        def allocate_vehicles(self, vehicle_list, *args, **kwargs):
            return {veh[0]: 0 for veh in vehicle_list}

        def update_edge_status(self, *args, **kwargs):
            pass

        def update_traffic_data(self, current_occupancy):
            self.history_traffic[:-1] = self.history_traffic[1:]
            self.history_traffic[-1] = current_occupancy
            self._replay_buffer.append(current_occupancy.copy())
            if len(self._replay_buffer) > self.replay_capacity:
                self._replay_buffer.pop(0)





    class DCRNN_BaselineAllocator:
        def __init__(self, num_edges, od_candidate_paths, adjacency_matrix, congestion_threshold=0.4, seq_len=12,
                     pred_len=1,
                     update_interval=1, scaler_mean=None, scaler_std=None, pretrained_path=None,
                     train_interval=20, train_batch_size=8, replay_capacity=600, learning_rate=1e-3):
            self.num_edges = num_edges
            self.congestion_threshold = congestion_threshold
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.update_interval = update_interval
            self.step_counter = 0

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if adjacency_matrix.shape[0] != self.num_edges:
                adjacency_matrix = adjacency_matrix[:self.num_edges, :self.num_edges]

            adj_sparse = build_adj_matrix(adjacency_matrix)
            self.forward_trans, self.backward_trans = compute_diffusion_matrices(adj_sparse, K=3)


            self.model = DCRNN(
                num_nodes=num_edges,
                input_dim=1,
                hidden_dim=16,
                output_dim=1,
                seq_len=seq_len,
                pred_len=pred_len,
                K=3,
                num_layers=2,
                forward_trans=self.forward_trans,
                backward_trans=self.backward_trans,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std
            ).to(self.device)

            if pretrained_path is not None:
                self.model.load_model(pretrained_path, device=self.device)
            self.model.eval()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
            self.train_interval = train_interval
            self.train_batch_size = train_batch_size
            self.replay_capacity = replay_capacity
            self._replay_buffer = []

            self.history_traffic = np.zeros((seq_len, num_edges), dtype=np.float32)
            init_val = scaler_mean if scaler_mean is not None else congestion_threshold
            self._cached_risk = np.full(num_edges, init_val, dtype=np.float32)

            self._path_priority_cache = {}
            self._path_edge_ids_cache = {}
            self._path_edge_pos_cache = {}
            self._path_edges_cache = {}
            self._path_offsets_cache = {}
            self._path_congest_cache = {}
            self._path_reduce_idx_cache = {}

        def set_edge_capacity(self, capacity_array):
            self.capacity = capacity_array

        def precompute_path_data(self, *args, **kwargs):
            pass

        def step_advance(self):
            self.step_counter += 1

        def calculate_congestion_risk(self):
            if self.step_counter % self.train_interval == 0:
                self._online_train()
            if self.step_counter % self.update_interval == 0:
                self._predict_congestion()
            return self._cached_risk

        def _online_train(self):
            needed = self.seq_len + self.pred_len
            n = len(self._replay_buffer)
            if n < needed + self.train_batch_size:
                return

            starts = np.random.randint(0, n - needed + 1, size=self.train_batch_size)
            x_batch, y_batch = [], []
            for s in starts:
                window = self._replay_buffer[s:s + needed]
                x_batch.append(np.stack(window[:self.seq_len], axis=0))
                y_batch.append(np.stack(window[self.seq_len:needed], axis=0))

            x_tensor = torch.from_numpy(np.stack(x_batch)[..., None]).float().to(self.device)
            y_tensor = torch.from_numpy(np.stack(y_batch)[..., None]).float().to(self.device)

            self.model.train()
            with torch.enable_grad():
                self.optimizer.zero_grad()
                pred = self.model(x_tensor, targets=y_tensor, training=True, sampling_prob=0.5)
                loss = self.criterion(pred, y_tensor)
                loss.backward()
                self.optimizer.step()
            self.model.eval()

        def _predict_congestion(self):
            with torch.no_grad():

                x = self.history_traffic[None, ..., None]
                x = torch.from_numpy(x).float().to(self.device)

                pred = self.model(x, training=False)

                self._cached_risk = pred.cpu().numpy().squeeze()
                self._cached_risk = np.clip(self._cached_risk, 0.0, REAL_OCC_MAX * 1.5)

        def allocate_vehicles(self, vehicle_list, *args, **kwargs):
            return {veh[0]: 0 for veh in vehicle_list}

        def update_edge_status(self, *args, **kwargs):
            pass

        def update_traffic_data(self, current_occupancy):
            self.history_traffic[:-1] = self.history_traffic[1:]
            self.history_traffic[-1] = current_occupancy
            self._replay_buffer.append(current_occupancy.copy())
            if len(self._replay_buffer) > self.replay_capacity:
                self._replay_buffer.pop(0)




    class STResNet_BaselineAllocator:
        def __init__(self, num_edges, od_candidate_paths, adjacency_matrix, congestion_threshold=0.4, seq_len=12,
                     pred_len=1,
                     update_interval=1, scaler_min=None, scaler_max=None, pretrained_path=None,
                     train_interval=20, train_batch_size=8, replay_capacity=600, learning_rate=1e-3):
            self.num_edges = num_edges
            self.congestion_threshold = congestion_threshold
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.update_interval = update_interval
            self.step_counter = 0

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.grid_height = 1
            self.grid_width = num_edges
            self.closeness_len = 3
            self.period_len = 1
            self.trend_len = 1


            self.model = STResNet(
                height=self.grid_height,
                width=self.grid_width,
                closeness_len=self.closeness_len,
                period_len=self.period_len,
                trend_len=self.trend_len,
                num_res_units=2,
                use_bn=True,
                scaler_min=scaler_min,
                scaler_max=scaler_max
            ).to(self.device)

            if pretrained_path is not None:
                self.model.load_model(pretrained_path, device=self.device)
            self.model.eval()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
            self.train_interval = train_interval
            self.train_batch_size = train_batch_size
            self.replay_capacity = replay_capacity
            self._replay_buffer = []

            self.history_traffic = np.zeros((seq_len, num_edges), dtype=np.float32)
            init_val = (scaler_min + scaler_max) / 2 if scaler_min is not None else congestion_threshold
            self._cached_risk = np.full(num_edges, init_val, dtype=np.float32)

            self._path_priority_cache = {}
            self._path_edge_ids_cache = {}
            self._path_edge_pos_cache = {}
            self._path_edges_cache = {}
            self._path_offsets_cache = {}
            self._path_congest_cache = {}
            self._path_reduce_idx_cache = {}

        def set_edge_capacity(self, capacity_array):
            self.capacity = capacity_array

        def precompute_path_data(self, *args, **kwargs):
            pass

        def step_advance(self):
            self.step_counter += 1

        def calculate_congestion_risk(self):
            if self.step_counter % self.train_interval == 0:
                self._online_train()
            if self.step_counter % self.update_interval == 0:
                self._predict_congestion()
            return self._cached_risk

        def _make_branch_input(self, window):
            arr = np.stack(window, axis=0)[None, :, None, :, :]
            return torch.from_numpy(arr).float().to(self.device)

        def _online_train(self):
            needed = self.closeness_len + self.pred_len
            n = len(self._replay_buffer)
            if n < needed + self.train_batch_size:
                return

            starts = np.random.randint(0, n - needed + 1, size=self.train_batch_size)
            c_batch, y_batch = [], []
            for s in starts:
                window = self._replay_buffer[s:s + needed]
                c_win = np.stack(window[:self.closeness_len], axis=0).reshape(self.closeness_len, self.grid_height, self.grid_width)
                y_win = window[self.closeness_len].reshape(self.grid_height, self.grid_width)
                c_batch.append(c_win)
                y_batch.append(y_win)

            c_tensor = torch.from_numpy(np.stack(c_batch)[:, :, None, :, :]).float().to(self.device)
            y_tensor = torch.from_numpy(np.stack(y_batch)[:, None, :, :]).float().to(self.device)
            p_tensor = torch.zeros(self.train_batch_size, self.period_len, 1, self.grid_height, self.grid_width, device=self.device)
            q_tensor = torch.zeros(self.train_batch_size, self.trend_len, 1, self.grid_height, self.grid_width, device=self.device)

            self.model.train()
            with torch.enable_grad():
                self.optimizer.zero_grad()
                pred = self.model(c_tensor, p_tensor, q_tensor, ext_x=None, return_original=False)
                loss = self.criterion(pred, y_tensor)
                loss.backward()
                self.optimizer.step()
            self.model.eval()

        def _predict_congestion(self):
            with torch.no_grad():

                closeness_x = self._make_branch_input(
                    [self.history_traffic[-self.closeness_len + i].reshape(self.grid_height, self.grid_width)
                     for i in range(self.closeness_len)]
                )
                period_x = torch.zeros(1, self.period_len, 1, self.grid_height, self.grid_width, device=self.device)
                trend_x = torch.zeros(1, self.trend_len, 1, self.grid_height, self.grid_width, device=self.device)


                pred = self.model(closeness_x, period_x, trend_x, return_original=True)
                pred_np = pred.cpu().numpy().squeeze()

                self._cached_risk = pred_np
                self._cached_risk = np.clip(self._cached_risk, 0.0, REAL_OCC_MAX * 1.5)

        def allocate_vehicles(self, vehicle_list, *args, **kwargs):
            return {veh[0]: 0 for veh in vehicle_list}

        def update_edge_status(self, *args, **kwargs):
            pass

        def update_traffic_data(self, current_occupancy):
            self.history_traffic[:-1] = self.history_traffic[1:]
            self.history_traffic[-1] = current_occupancy
            self._replay_buffer.append(current_occupancy.copy())
            if len(self._replay_buffer) > self.replay_capacity:
                self._replay_buffer.pop(0)





    NETWORK_NAME = "Philadelphia"
    NODE_FILE = f"{NETWORK_NAME}.nod.xml"
    EDGE_FILE = f"{NETWORK_NAME}.edg.xml"
    CAPACITY_CSV = f"edge_capacity_{NETWORK_NAME.lower()}.csv"
    CANDIDATE_PKL = f"od_candidate_paths_{NETWORK_NAME.lower()}.pkl"
    TRIPS_FILE = f"{NETWORK_NAME}_sorted.trips.xml"
    NETCONVERT_EXE = r"D:\安装程序\sumo-win64-1.27.1\sumo-1.27.1\bin\netconvert.exe"

    SIM_DURATION = 1800
    SEED = 42

    NETWORK_SCALE_RATIO = 0.2
    TARGET_TRIP_COUNT = 1_000_000

    net = None

    print(f"⏳ 正在流式扫描路网节点文件...")
    node_in_scope, node_coords, node_types = stream_load_node_bbox_and_scope(NODE_FILE, NETWORK_SCALE_RATIO)
    print(f"✅ 范围内节点数: {len(node_in_scope)}")

    print(f"⏳ 正在流式扫描路网边文件...")
    edges, edge_from_node, edge_to_node, edge_num_lanes, _ffs, _elens, required_node_ids = stream_load_cropped_edges(EDGE_FILE, node_in_scope)
    edge_id_to_idx = {eid: i for i, eid in enumerate(edges)}
    edge_idx_to_id = {i: eid for i, eid in enumerate(edges)}
    num_edges = len(edges)
    free_flow_speeds = np.array(_ffs, dtype=np.float32)
    edge_lengths = np.array(_elens, dtype=np.float32)
    print(f"✅ 范围内边数: {num_edges}, 所需节点数: {len(required_node_ids)}")

    adj_matrix = build_adjacency_matrix(edges, edge_from_node, edge_to_node, edge_id_to_idx)

    import os
    NET_FILE = f"{NETWORK_NAME}_cropped_{NETWORK_SCALE_RATIO}.net.xml"
    if not os.path.exists(NET_FILE):
        print(f"⏳ 正在使用netconvert编译裁剪路网...")
        build_cropped_sumo_network(
            NETCONVERT_EXE, NET_FILE, node_coords, node_types,
            edges, edge_from_node, edge_to_node, edge_num_lanes,
            free_flow_speeds, edge_lengths, required_node_ids
        )
        print(f"✅ 生成裁剪路网文件: {NET_FILE}")
    else:
        print(f"✅ 使用已有裁剪路网文件: {NET_FILE}")

    print(f"⏳ 正在计算裁剪路网真实可达性...")
    edge_out_connections = build_edge_connectivity_from_net(NET_FILE)
    edge_reachable_nodes = build_edge_reachable_nodes(edges, edge_out_connections, edge_to_node)
    junction_out_edges = defaultdict(list)
    for eid, n in edge_from_node.items():
        junction_out_edges[n].append(eid)
    print(f"✅ 可达性计算完成")

    filtered_trips = f"{NETWORK_NAME}_filtered_{NETWORK_SCALE_RATIO}_{SIM_DURATION}_{TARGET_TRIP_COUNT}_v4.trips.xml"

    if not os.path.exists(filtered_trips):
        print(f"⏳ 正在统计范围内可用行程数...")
        eligible_count = count_eligible_trips(TRIPS_FILE, required_node_ids, edge_reachable_nodes, edge_id_to_idx, junction_out_edges, SIM_DURATION)
        sample_prob = min(1.0, TARGET_TRIP_COUNT / max(eligible_count, 1))
        print(f"✅ 范围内可用行程数: {eligible_count}, 采样概率: {sample_prob:.4f}")

        print(f"⏳ 正在流式采样并生成裁剪trips文件...")
        _veh_pre_ids, _veh_pre_rows = stream_sample_trips(
            TRIPS_FILE, filtered_trips, required_node_ids, edge_reachable_nodes, edge_from_node,
            edge_id_to_idx, SIM_DURATION, sample_prob, SEED
        )
        print(f"✅ 生成裁剪trips文件: {filtered_trips}")
    else:
        print(f"✅ 使用已有裁剪trips文件: {filtered_trips}")
        _veh_pre_ids, _veh_pre_rows = parse_filtered_trips(filtered_trips, edge_id_to_idx, required_node_ids, edge_reachable_nodes)

    TRIPS_FILE = filtered_trips

    veh_pre_data = np.array(_veh_pre_rows, dtype=VEH_PRE_MAP_DTYPE)
    veh_pre_index = {vid: i for i, vid in enumerate(_veh_pre_ids)}
    veh_pre_map = VehPreMap(veh_pre_index, veh_pre_data)
    print(f"✅ 预加载 {len(veh_pre_map)} 条行程")

    demand_od_set = {(int(row[0]), int(row[1])) for row in _veh_pre_rows}
    print(f"✅ 行程覆盖OD对数: {len(demand_od_set)}")

    cap_df = pandas.read_csv(CAPACITY_CSV, index_col=0)
    capacity_array = np.zeros(num_edges, dtype=np.float32)
    for edge_id, row in cap_df.iterrows():
        if edge_id in edge_id_to_idx:
            capacity_array[edge_id_to_idx[edge_id]] = row["capacity_veh_per_hour"]

    print(f"⏳ 正在加载od_candidate_paths...")
    import time
    t_start = time.time()
    with open(CANDIDATE_PKL, "rb") as f:
        data = pickle.load(f)
    _raw_od_candidate_paths = data["od_candidate_paths"]
    pickle_edge_id_to_idx = data.get("edge_id_to_idx", {})
    pickle_idx_to_edge_id = {v: k for k, v in pickle_edge_id_to_idx.items()}
    print(f"  pickle加载耗时: {time.time()-t_start:.2f}秒")

    top_od_set = demand_od_set

    print(f"⏳ 正在重映射路径索引...")
    t_start = time.time()
    od_candidate_paths = {}
    for od_key, paths in _raw_od_candidate_paths.items():
        if top_od_set is not None and od_key not in top_od_set:
            continue
        valid_paths = []
        for path in paths:
            path_arr = np.asarray(path, dtype=np.int32)
            remapped_path = []
            valid = True
            for pickle_idx in path_arr:
                if pickle_idx in pickle_idx_to_edge_id:
                    edge_id = pickle_idx_to_edge_id[pickle_idx]
                    if edge_id in edge_id_to_idx:
                        remapped_path.append(edge_id_to_idx[edge_id])
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid and len(remapped_path) > 0:
                valid_paths.append(np.array(remapped_path, dtype=np.int32))
        if len(valid_paths) > 0:
            od_candidate_paths[od_key] = valid_paths
    del data, _raw_od_candidate_paths
    print(f"  路径重映射耗时: {time.time()-t_start:.2f}秒")
    print(f"✅ 有效OD对: {len(od_candidate_paths)}, 总路径数: {sum(len(p) for p in od_candidate_paths.values())}")


    threshold_quantiles = [
        # 0.8,
        # 0.9,
        0.95,
        # 0.99,
    ]

    def _build_threshold_probe_allocator(loose_threshold):
        print(f"⏳ 正在构建allocator (threshold={loose_threshold:.3f})...")
        t_start = time.time()
        probe_alloc = EndogenousRouteAllocator(
            num_edges=num_edges,
            od_candidate_paths=od_candidate_paths,
            congestion_threshold=loose_threshold,
            smooth_window=60,
            time_decay_factor=0.95,
            congestion_update_interval=1,
            use_d3stn_enhance=False,
        )
        probe_alloc.set_edge_capacity(
            capacity_array * probe_alloc.congestion_tracker.time_window_size / 3600 * 0.08
        )
        print(f"  allocator初始化耗时: {time.time()-t_start:.2f}秒")
        print(f"⏳ 正在预计算路径数据...")
        t_start = time.time()
        probe_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)
        print(f"  路径预计算耗时: {time.time()-t_start:.2f}秒")
        return probe_alloc

    # 注：threshold_list（主实验要扫的阈值候选）依赖 Calibration 阶段采集的真实占有率样本，
    # 主实验（下面的 for thresh in threshold_list 主对比表）离不开它，因此本次"只跑主实验"
    # 时仍保留这部分采样运行；真正被注释停用的是下面的 Validation(12轮)+Test(1轮) 探针仿真
    # 和汇总打印（那是另一套独立的"冻结阈值"报告，不影响 threshold_list / 主对比表）。
    calibration_occupancy_samples = []
    for calib_seed in CALIBRATION_SEEDS:
        calib_alloc = _build_threshold_probe_allocator(REAL_OCC_MIN + 0.5 * (REAL_OCC_MAX - REAL_OCC_MIN))
        _, _, _, calib_net, _ = run_single_simulation(
            "Calibration", calib_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, calib_seed, f"tripinfo_calibration_{calib_seed}.xml"
        )
        calibration_occupancy_samples.append(calib_net["real_occupancy_history"].flatten())
    calibration_occupancy_samples = np.concatenate(calibration_occupancy_samples)

    nonzero_occupancy_samples = calibration_occupancy_samples[calibration_occupancy_samples > 0]
    if nonzero_occupancy_samples.size == 0:
        print("⚠️ 校准样本中没有非零占有率读数，回退使用全量样本计算分位数")
        nonzero_occupancy_samples = calibration_occupancy_samples
    print(
        f"✅ 校准样本过滤: 总样本数={calibration_occupancy_samples.shape[0]}, "
        f"有交通压力(非零)样本数={nonzero_occupancy_samples.shape[0]} "
        f"({100.0 * nonzero_occupancy_samples.shape[0] / max(calibration_occupancy_samples.shape[0], 1):.2f}%)"
    )

    threshold_list = [float(np.quantile(nonzero_occupancy_samples, q)) for q in threshold_quantiles]
    print(f"✅ 过滤后的阈值候选(threshold_list): {[f'{t:.6f}' for t in threshold_list]}")

    # ==================== Validation / Test 阶段：阈值分位数选择 ====================
    # 该阶段用检测精度/召回（触发边计数 vs 实际拥堵边计数）来挑选 threshold_quantiles 中最优的
    # 分位数 q，属于"阈值本身如何标定"的内部指标，与主对比表里 Original/D3STN 等算法的
    # Precision/Recall 列（已从 Control Performance Table 中移除）是两回事，因此保留。
    #
    # 修复说明：原来的写法是对每个 validation seed 各自算一次 precision 再取平均，如果某个候选
    # 阈值只触发了1~2条边，该seed的precision很容易凑巧是100%，却和触发了几百条边的seed的
    # precision等权重平均，从而把整体均值拉高，造出"precision=1.0000"的假象。修复思路：改成
    # 对多个validation seed的原始命中数/触发数做pooled求和后再算一次比例，并加一个最小触发边数
    # 门槛，触发边数太少的候选直接判定为"统计不可信"而不参与比较。

    MIN_TRIGGERED_EDGES_FOR_TRUST = 30

    def select_frozen_threshold(candidate_q_values):
        best_q = candidate_q_values[0]
        best_score = -1.0
        best_recall = -1.0
        best_precision = -1.0
        for q in candidate_q_values:
            candidate_threshold = float(np.quantile(nonzero_occupancy_samples, q))
            pooled_precision_hits = 0
            pooled_triggered_edges = 0
            pooled_recall_hits = 0
            pooled_congested_edges = 0
            for val_seed in VALIDATION_SEEDS:
                val_alloc = _build_threshold_probe_allocator(candidate_threshold)
                val_metrics, _, _, _, _ = run_single_simulation(
                    "Validation", val_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
                    free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
                    TRIPS_FILE, SIM_DURATION, val_seed, f"tripinfo_validation_q{q}_{val_seed}.xml"
                )
                pooled_precision_hits += val_metrics["precision_hit_count"]
                pooled_triggered_edges += val_metrics["triggered_edge_count"]
                pooled_recall_hits += val_metrics["recall_hit_count"]
                pooled_congested_edges += val_metrics["congested_edge_count"]
            # 用跨seed汇总(pooled)后的原始计数算precision/recall，而不是先各自算比例再平均，
            # 避免触发样本极少的seed噪声被等权重拉入均值
            mean_precision = pooled_precision_hits / pooled_triggered_edges if pooled_triggered_edges > 0 else 0.0
            mean_recall = pooled_recall_hits / pooled_congested_edges if pooled_congested_edges > 0 else 0.0
            if pooled_triggered_edges < MIN_TRIGGERED_EDGES_FOR_TRUST:
                print(
                    f"  q={q}: 触发边数仅{pooled_triggered_edges} < {MIN_TRIGGERED_EDGES_FOR_TRUST}，"
                    f"precision={mean_precision:.4f}视为小样本假象，跳过该候选"
                )
                continue
            score = (2 * mean_precision * mean_recall / (mean_precision + mean_recall)
                     if (mean_precision + mean_recall) > 0 else 0.0)
            if score > best_score:
                best_score = score
                best_recall = mean_recall
                best_precision = mean_precision
                best_q = q
        return float(np.quantile(nonzero_occupancy_samples, best_q)), best_q, best_recall, best_precision

    frozen_threshold, frozen_q, frozen_val_recall, frozen_val_precision = select_frozen_threshold(
        threshold_quantiles)

    test_recalls = []
    test_precisions = []
    for test_seed in TEST_SEEDS:
        frozen_alloc = _build_threshold_probe_allocator(frozen_threshold)
        frozen_test_metrics, _, _, _, _ = run_single_simulation(
            "Test", frozen_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, test_seed, f"tripinfo_test_frozen_{test_seed}.xml"
        )
        test_recalls.append(frozen_test_metrics["recall"])
        test_precisions.append(frozen_test_metrics["precision"])

    print("\n" + "=" * 100)
    print("Calibration -> Validation -> Test Threshold Selection")
    print("=" * 100)
    print(f"Calibration seeds: {CALIBRATION_SEEDS}, samples collected: {calibration_occupancy_samples.shape[0]}")
    print(
            f"Validation seeds: {VALIDATION_SEEDS}, selected quantile q = {frozen_q}, "
            f"mean validation recall = {frozen_val_recall:.4f}, mean validation precision = {frozen_val_precision:.4f}"
        )
    print(f"Frozen threshold (from calibration quantile) = {frozen_threshold:.6f}")
    print(
            f"Test seeds: {TEST_SEEDS}, mean test recall at frozen threshold = {float(np.mean(test_recalls)):.4f}, "
            f"mean test precision at frozen threshold = {float(np.mean(test_precisions)):.4f}"
        )
    print("=" * 100)

    w = 18



    def pct(v):
        return f"{v * 100:.2f}%"

    def _kp_label(feedback_gain, base_name):
        if feedback_gain is not None and feedback_gain <= 1e-6:
            return "Detection-only (open-loop, no actuation)"
        return base_name

    def _kp_condition_label(feedback_gain, default_label):
        if feedback_gain is not None and feedback_gain <= 1e-6:
            return "Free-flow weighted baseline"
        return default_label

    def print_control_performance_table(thresh, control_rows):
        print(f"\n{'=' * 120}")
        print(f"Traffic loading model: {TRAFFIC_LOADING_MODEL}")
        print(f"Congestion threshold under test: congestion_threshold = {thresh:.4f}")
        print("=" * 120)

        labels = [_kp_label(fg, name) for name, m, net_m, eff_m, fg in control_rows]
        conditions = [_kp_condition_label(fg, "Closed-loop feedback control") for name, m, net_m, eff_m, fg in control_rows]

        print("\n" + "=" * 120)
        print("Control Performance Table (Original Endogenous / D3STN-Enhanced / Marginal-Cost / MPC)")
        print("=" * 120)
        header = f"{'Metric':<32}"
        for lbl in labels:
            header += f"{lbl:<{w}}"
        print(header)
        cond_line = f"{'Control condition':<32}"
        for cond in conditions:
            cond_line += f"{cond:<{w}}"
        print(cond_line)
        print("-" * 120)

        def row(label, key, fmt, src="m"):
            line = f"{label:<32}"
            for name, m, net_m, eff_m, fg in control_rows:
                d = m if src == "m" else (net_m if src == "net" else eff_m)
                val = d[key]
                line += f"{fmt(val):<{w}}"
            print(line)

        row("Avg Travel Time (s)", "avg_trip_time", lambda v: f"{v:.1f}", src="net")
        row("Total Congestion Time (edge*s)", "total_congestion_time", lambda v: f"{v:.0f}")
        row("Avg Congestion Depth", "avg_congestion_depth", lambda v: f"{v:.4f}")
        row("Avg Congestion Duration (s)", "real_avg_congestion_duration", lambda v: f"{v:.1f}")
        row("Avg Recovery Time (s)", "avg_recovery_time", lambda v: f"{v:.1f}")
        row("Occupancy Std Dev", "real_occupancy_std", lambda v: f"{v:.4f}")
        row("Computation Latency (CPU, ms/step)", "avg_step_ms", lambda v: f"{v:.2f}", src="eff")
        print("=" * 120)

        travel_times = [(name, net_m["avg_trip_time"]) for name, m, net_m, eff_m, fg in control_rows]
        best_name, best_time = min(travel_times, key=lambda x: x[1])
        worst_name, worst_time = max(travel_times, key=lambda x: x[1])
        print(
            f"Lowest average travel time: {best_name} ({best_time:.1f}s), "
            f"highest: {worst_name} ({worst_time:.1f}s)"
        )

        has_d3stn = any("D" in name and "STN" in name for name, m, net_m, eff_m, fg in control_rows)
        if has_d3stn:
            print(
                "Coupling note: D3STN-Enhanced's congestion_risk, produced by "
                "congestion_tracker.calculate_congestion_risk(), is passed directly into allocate_vehicles()' "
                "path-reduction logic, so its route diversion decisions are conditioned on the D3STN "
                "spatial-temporal risk signal rather than on raw occupancy alone."
            )
        print("=" * 120)

    def print_prediction_accuracy_table(thresh, predictor_rows):
        print(f"\n{'=' * 120}")
        print(f"Traffic loading model: {TRAFFIC_LOADING_MODEL}")
        print(f"Congestion threshold under test: congestion_threshold = {thresh:.4f}")
        print("=" * 120)
        print("Prediction Accuracy Table (STGCN / DCRNN / ST-ResNet)")
        print(
            "These strategies are open-loop occupancy predictors whose allocate_vehicles() returns a no-op "
            "assignment; no travel-time or control-performance columns are reported here since they do not "
            "actuate any route diversion."
        )
        print("-" * 120)
        header = f"{'Metric':<32}"
        for name, m, net_m, eff_m in predictor_rows:
            header += f"{name:<{w}}"
        print(header)
        print("-" * 120)

        def row(label, key, fmt, src="m"):
            line = f"{label:<32}"
            for name, m, net_m, eff_m in predictor_rows:
                d = m if src == "m" else (net_m if src == "net" else eff_m)
                line += f"{fmt(d[key]):<{w}}"
            print(line)

        row("MAE", "raw_mae", lambda v: f"{v:.4f}")
        row("RMSE", "raw_rmse", lambda v: f"{v:.4f}")
        row("Top-10% Overlap", "top_hit_ratio", pct)
        row("Computation Latency (CPU, ms/step)", "avg_step_ms", lambda v: f"{v:.2f}", src="eff")
        print("=" * 120)


    all_threshold_results = []

    for thresh in threshold_list:
        print(f"\n\n{'=' * 100}")
        print(f"🚀 开始测试：拥堵阈值 = {thresh:.4f}")
        print("=" * 100)


        cap_percentiles = np.percentile(capacity_array, [20, 40, 60, 80])
        edgewise_threshold = np.full(num_edges, thresh, dtype=np.float32)

        high_mask = capacity_array >= cap_percentiles[3]
        edgewise_threshold[high_mask] = thresh * 1.3
        mid_high_mask = (capacity_array >= cap_percentiles[2]) & (capacity_array < cap_percentiles[3])
        edgewise_threshold[mid_high_mask] = thresh * 1.15
        mid_low_mask = (capacity_array >= cap_percentiles[1]) & (capacity_array < cap_percentiles[2])
        edgewise_threshold[mid_low_mask] = thresh * 0.85
        low_mask = capacity_array < cap_percentiles[1]
        edgewise_threshold[low_mask] = thresh * 0.7
        edgewise_threshold = np.clip(edgewise_threshold, a_min=REAL_OCC_MIN*0.5, a_max=REAL_OCC_MAX*1.2).astype(np.float32)


        print("\n运行【原版内生策略】...")
        base_alloc = EndogenousRouteAllocator(
            num_edges=num_edges,
            od_candidate_paths=od_candidate_paths,
            congestion_threshold=edgewise_threshold,
            smooth_window=60,
            time_decay_factor=0.95,
            congestion_update_interval=1,
            use_d3stn_enhance=False,
        )
        base_alloc.set_edge_capacity(
            capacity_array * base_alloc.congestion_tracker.time_window_size / 3600 * 0.08
        )
        base_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)

        base_metrics, base_lead_time, base_assigned, base_net, base_eff = run_single_simulation(
            "原版内生策略", base_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_base_{thresh:.4f}.xml"
        )


        print("\n运行【D³STN】...")
        d3stn_alloc = EndogenousRouteAllocator(
            num_edges=num_edges,
            od_candidate_paths=od_candidate_paths,
            congestion_threshold=edgewise_threshold,
            smooth_window=60,
            time_decay_factor=0.95,
            congestion_update_interval=1,
            use_d3stn_enhance=True,
        )
        d3stn_alloc.congestion_tracker.enable_dynamic_tau = False
        d3stn_alloc.set_edge_capacity(
            capacity_array * d3stn_alloc.congestion_tracker.time_window_size / 3600 * 0.08
        )
        d3stn_alloc.init_d3stn_gnn(adj_matrix, edge_lengths, free_flow_speeds)
        d3stn_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)

        d3stn_metrics, d3stn_lead_time, d3stn_assigned, d3stn_net, d3stn_eff = run_single_simulation(
            "D³STN", d3stn_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_d3stn_{thresh:.4f}.xml"
        )


        print("\n运行【STGCN时空图卷积基线】...")
        stgcn_alloc = STGCN_BaselineAllocator(
            num_edges=num_edges,
            od_candidate_paths=od_candidate_paths,
            adjacency_matrix=adj_matrix,
            congestion_threshold=thresh,
            update_interval=1,
            scaler_mean=REAL_OCC_MEAN,
            scaler_std=REAL_OCC_STD,
            pretrained_path=None
        )
        stgcn_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)

        stgcn_metrics, stgcn_lead_time, stgcn_assigned, stgcn_net, stgcn_eff = run_single_simulation(
            "STGCN基线策略", stgcn_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_stgcn_{thresh:.4f}.xml"
        )


        print("\n运行【DCRNN扩散卷积基线】...")
        dcrnn_alloc = DCRNN_BaselineAllocator(
            num_edges=num_edges,
            od_candidate_paths=od_candidate_paths,
            adjacency_matrix=adj_matrix,
            congestion_threshold=thresh,
            update_interval=1,
            scaler_mean=REAL_OCC_MEAN,
            scaler_std=REAL_OCC_STD,
            pretrained_path=None
        )
        dcrnn_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)

        dcrnn_metrics, dcrnn_lead_time, dcrnn_assigned, dcrnn_net, dcrnn_eff = run_single_simulation(
            "DCRNN基线策略", dcrnn_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_dcrnn_{thresh:.4f}.xml"
        )


        print("\n运行【ST-ResNet时空残差基线】...")
        stresnet_alloc = STResNet_BaselineAllocator(
            num_edges=num_edges,
            od_candidate_paths=od_candidate_paths,
            adjacency_matrix=adj_matrix,
            congestion_threshold=thresh,
            update_interval=1,
            scaler_min=REAL_OCC_MIN,
            scaler_max=REAL_OCC_MAX,
            pretrained_path=None
        )
        stresnet_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)

        stresnet_metrics, stresnet_lead_time, stresnet_assigned, stresnet_net, stresnet_eff = run_single_simulation(
            "ST-ResNet基线策略", stresnet_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_stresnet_{thresh:.4f}.xml"
        )


        print("\n运行【Marginal-Cost Routing】...")
        mc_alloc = MarginalCostRoutingAllocator(
            num_edges=num_edges,
            od_candidate_paths=od_candidate_paths,
            congestion_threshold=edgewise_threshold,
            smooth_window=60,
            time_decay_factor=0.95,
            congestion_update_interval=1,
            use_d3stn_enhance=False,
            gamma_max=1.0,
        )
        mc_alloc.set_edge_capacity(
            capacity_array * mc_alloc.congestion_tracker.time_window_size / 3600 * 0.08
        )
        mc_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)

        mc_metrics, mc_lead_time, mc_assigned, mc_net, mc_eff = run_single_simulation(
            "Marginal-Cost Routing", mc_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_marginalcost_{thresh:.4f}.xml"
        )


        print("\n运行【MPC Routing】...")
        mpc_alloc = MPCRoutingAllocator(
            num_edges=num_edges,
            od_candidate_paths=od_candidate_paths,
            congestion_threshold=edgewise_threshold,
            smooth_window=60,
            time_decay_factor=0.95,
            congestion_update_interval=1,
            use_d3stn_enhance=False,
            gamma_max=1.0,
        )
        mpc_alloc.set_edge_capacity(
            capacity_array * mpc_alloc.congestion_tracker.time_window_size / 3600 * 0.08
        )
        mpc_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)

        mpc_metrics, mpc_lead_time, mpc_assigned, mpc_net, mpc_eff = run_single_simulation(
            "MPC Routing", mpc_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
            free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
            TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_mpc_{thresh:.4f}.xml"
        )


        print_control_performance_table(thresh, [
            ("Original Endogenous", base_metrics, base_net, base_eff, base_alloc.feedback_gain),
            ("D3STN-Enhanced", d3stn_metrics, d3stn_net, d3stn_eff, d3stn_alloc.feedback_gain),
            ("Marginal-Cost Routing", mc_metrics, mc_net, mc_eff, getattr(mc_alloc, "feedback_gain", None)),
            ("MPC Routing", mpc_metrics, mpc_net, mpc_eff, getattr(mpc_alloc, "feedback_gain", None)),
        ])

        print_prediction_accuracy_table(thresh, [
            ("STGCN Baseline", stgcn_metrics, stgcn_net, stgcn_eff),
            ("DCRNN Baseline", dcrnn_metrics, dcrnn_net, dcrnn_eff),
            ("ST-ResNet Baseline", stresnet_metrics, stresnet_net, stresnet_eff),
        ])

        thresh_all_metrics = []
        for m, net_m, eff_m in zip(
                [base_metrics, d3stn_metrics, stgcn_metrics, dcrnn_metrics, stresnet_metrics, mc_metrics, mpc_metrics],
                [base_net, d3stn_net, stgcn_net, dcrnn_net, stresnet_net, mc_net, mpc_net],
                [base_eff, d3stn_eff, stgcn_eff, dcrnn_eff, stresnet_eff, mc_eff, mpc_eff]
        ):

            merged = {**m, **net_m, **eff_m}
            merged.pop("real_occupancy_history", None)
            merged.pop("lead_time_distribution", None)
            thresh_all_metrics.append(merged)
        all_threshold_results.append(thresh_all_metrics)


        plot_bar_single_threshold(thresh, thresh_all_metrics, save_path=f"bar_threshold_{thresh:.4f}.pdf")


    plot_trend_all_thresholds(threshold_list, all_threshold_results, save_path="metric_trend_all.pdf")

    print("\n" + "=" * 100)
    print("Threshold Sensitivity Analysis (rate of decline of diversion-trigger ratio; higher = more sensitive)")
    print("=" * 100)
    for s_idx, s_name in enumerate(STRATEGY_NAMES):
        trigger_ratios = [all_threshold_results[t_idx][s_idx]['business_trigger_ratio'] for t_idx in
                          range(len(threshold_list))]
        sensitivity = calculate_threshold_sensitivity(threshold_list, trigger_ratios)
        print(f"{s_name:<24} sensitivity: {sensitivity:.4f} / unit threshold")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("Hard-Saturation Comparison (gamma_max=0.4)")
    print("=" * 100)
    hard_sat_thresh = threshold_list[len(threshold_list) // 2]
    hard_cap_percentiles = np.percentile(capacity_array, [20, 40, 60, 80])
    hard_edgewise_threshold = np.full(num_edges, hard_sat_thresh, dtype=np.float32)
    hard_high_mask = capacity_array >= hard_cap_percentiles[3]
    hard_edgewise_threshold[hard_high_mask] = hard_sat_thresh * 1.3
    hard_mid_high_mask = (capacity_array >= hard_cap_percentiles[2]) & (capacity_array < hard_cap_percentiles[3])
    hard_edgewise_threshold[hard_mid_high_mask] = hard_sat_thresh * 1.15
    hard_mid_low_mask = (capacity_array >= hard_cap_percentiles[1]) & (capacity_array < hard_cap_percentiles[2])
    hard_edgewise_threshold[hard_mid_low_mask] = hard_sat_thresh * 0.85
    hard_low_mask = capacity_array < hard_cap_percentiles[1]
    hard_edgewise_threshold[hard_low_mask] = hard_sat_thresh * 0.7
    hard_edgewise_threshold = np.clip(
        hard_edgewise_threshold, a_min=REAL_OCC_MIN * 0.5, a_max=REAL_OCC_MAX * 1.2
    ).astype(np.float32)

    hard_base_alloc = EndogenousRouteAllocator(
        num_edges=num_edges,
        od_candidate_paths=od_candidate_paths,
        congestion_threshold=hard_edgewise_threshold,
        smooth_window=60,
        time_decay_factor=0.95,
        congestion_update_interval=1,
        use_d3stn_enhance=False,
        gamma_max=0.4,
    )
    hard_base_alloc.set_edge_capacity(
        capacity_array * hard_base_alloc.congestion_tracker.time_window_size / 3600 * 0.08
    )
    hard_base_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)
    hard_base_metrics, hard_base_lead, hard_base_assigned, hard_base_net, hard_base_eff = run_single_simulation(
        "Original Endogenous (gamma_max=0.4)", hard_base_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
        free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
        TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_hardsat_base_{hard_sat_thresh:.4f}.xml"
    )

    hard_d3stn_alloc = EndogenousRouteAllocator(
        num_edges=num_edges,
        od_candidate_paths=od_candidate_paths,
        congestion_threshold=hard_edgewise_threshold,
        smooth_window=60,
        time_decay_factor=0.95,
        congestion_update_interval=1,
        use_d3stn_enhance=True,
        gamma_max=0.4,
    )
    hard_d3stn_alloc.congestion_tracker.enable_dynamic_tau = False
    hard_d3stn_alloc.set_edge_capacity(
        capacity_array * hard_d3stn_alloc.congestion_tracker.time_window_size / 3600 * 0.08
    )
    hard_d3stn_alloc.init_d3stn_gnn(adj_matrix, edge_lengths, free_flow_speeds)
    hard_d3stn_alloc.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)
    hard_d3stn_metrics, hard_d3stn_lead, hard_d3stn_assigned, hard_d3stn_net, hard_d3stn_eff = run_single_simulation(
        "D3STN-Enhanced (gamma_max=0.4)", hard_d3stn_alloc, net, edges, edge_id_to_idx, edge_idx_to_id,
        free_flow_speeds, edge_lengths, edge_from_node, veh_pre_map,
        TRIPS_FILE, SIM_DURATION, SEED, f"tripinfo_hardsat_d3stn_{hard_sat_thresh:.4f}.xml"
    )

    print_control_performance_table(hard_sat_thresh, [
        ("Original Endogenous", hard_base_metrics, hard_base_net, hard_base_eff, hard_base_alloc.feedback_gain),
        ("D3STN-Enhanced", hard_d3stn_metrics, hard_d3stn_net, hard_d3stn_eff, hard_d3stn_alloc.feedback_gain),
    ])

    print("\n" + "=" * 100)
    print("Sensitivity Sweep: T_w (smooth_window) x alpha (time_decay_factor)")
    print("Kp (feedback_gain) sensitivity is covered separately in Kp.py's dedicated ablation study;")
    print("this sweep instead targets the estimator hyperparameters flagged as under-tested (Theorem 1).")
    print("=" * 100)

    # 使用默认阈值而不是依赖 hard_edgewise_threshold
    default_threshold = np.full(num_edges, 0.005, dtype=np.float32)

    sweep_base_config = {
        "num_edges": num_edges,
        "od_candidate_paths": od_candidate_paths,
        "allocator_kwargs": {
            "congestion_threshold": default_threshold,
            "congestion_update_interval": 1,
            "use_d3stn_enhance": False,
            "gamma_max": 0.7,
            "feedback_gain": 0.02,
        },
        "edge_capacity": capacity_array * 60 / 3600 * 0.08,
        "free_flow_speeds": free_flow_speeds,
        "edge_lengths": edge_lengths,
        "edge_idx_to_id": edge_idx_to_id,
        "net": net,
        "edges": edges,
        "edge_id_to_idx": edge_id_to_idx,
        "edge_from_node": edge_from_node,
        "veh_map": veh_pre_map,
        "trips_file": TRIPS_FILE,
        "sim_duration": SIM_DURATION,
        "seed": SEED,
        "output_prefix": "tripinfo_sweep",
    }
    sweep_results = run_2axis_sensitivity_sweep(
        "smooth_window", [15, 30, 60, 120],
        "time_decay_factor", [0.85, 0.90, 0.95, 0.99],
        sweep_base_config
    )
    for _, sweep_row in sweep_results.iterrows():
        print(
            f"T_w={sweep_row['smooth_window_value']}, alpha={sweep_row['param_value']}: "
            f"total_congestion_time={sweep_row['total_congestion_time']:.0f}, "
            f"trigger_ratio={sweep_row['business_trigger_ratio']:.4f}"
        )
    print("=" * 100)
    print("=" * 100)