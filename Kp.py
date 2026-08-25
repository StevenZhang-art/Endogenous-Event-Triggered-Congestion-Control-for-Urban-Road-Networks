import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from numba import jit
import gc
from collections import defaultdict
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
import xml.etree.ElementTree as ET
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.sparse as sp
from matplotlib import rcParams
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

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans']
rcParams['axes.unicode_minus'] = False
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['figure.dpi'] = 150

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
MARKERS = ['o', 's', '^', 'D', 'v', 'p']

TRAFFIC_LOADING_MODEL = "SUMO_MICROSCOPIC_CARFOLLOWING"
WARMUP_DURATION = 300
CLEARANCE_DURATION = 600
LEAD_TIME_MAX_WINDOW = 1800
CALIBRATION_SEEDS = [101, 102, 103]
VALIDATION_SEEDS = [201, 202, 203]
TEST_SEEDS = [42, 7, 13]
KP_VALUES = [0.00, 0.01, 0.02, 0.05]
GAMMA_MAX_PRIMARY = 0.7
GAMMA_MAX_HARD_SATURATION = 0.4
SUMO_EXE = r"D:\安装程序\sumo-win64-1.27.1\sumo-1.27.1\bin\sumo.exe"


class EndogenousCongestionTracker:
    def __init__(self,
                 num_edges,
                 congestion_threshold=0.4,
                 max_single_edge_ratio=0.2,
                 min_base_allocation=0.001,
                 time_window_size=60,
                 decay_factor=0.95,
                 max_global_congested_ratio=0.4,
                 gamma_max=None):
        self.num_edges = num_edges
        if np.isscalar(congestion_threshold):
            self.congestion_threshold = np.full(num_edges, congestion_threshold, dtype=np.float32)
        else:
            self.congestion_threshold = np.asarray(congestion_threshold, dtype=np.float32)
        self.max_single_edge_ratio = max_single_edge_ratio
        self.min_base_allocation = min_base_allocation
        self.time_window_size = time_window_size
        self.decay_factor = decay_factor
        self.gamma_max = max_global_congested_ratio if gamma_max is None else gamma_max
        self.max_global_congested_ratio = self.gamma_max

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
        decayed_weights = weights * self._decay_pows[offsets]

        flat_idx = edge_indices * self.time_window_size + target_indices
        aggregated = np.bincount(flat_idx, weights=decayed_weights, minlength=self._flat_total)
        self.allocation_windows += aggregated.reshape(self.num_edges, self.time_window_size)

        self.total_allocated_vehicles += np.sum(weights)

    def calculate_congestion_risk(self):
        window_total = np.sum(self.allocation_windows * self._decay_pows[None, :], axis=1)
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

    def get_global_congested_limit(self, total_demand):
        return total_demand * self.max_global_congested_ratio

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
                 max_global_congested_ratio=0.4,
                 gamma_max=None,
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

        self.edge_status = np.ones(num_edges, dtype=np.bool_)
        self.history_congestion_ratio = np.zeros((num_edges, smooth_window), dtype=np.float32)
        self.window_idx = 0

        self.feedback_gain = feedback_gain
        self.gamma_max = max_global_congested_ratio if gamma_max is None else gamma_max

        self.congestion_tracker = EndogenousCongestionTracker(
                num_edges=num_edges,
                congestion_threshold=congestion_threshold,
                max_single_edge_ratio=max_single_edge_ratio,
                min_base_allocation=min_base_allocation,
                time_window_size=smooth_window,
                decay_factor=time_decay_factor,
                max_global_congested_ratio=max_global_congested_ratio,
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

            high_congest_alloc = np.sum(path_allocation[path_has_congest])
            high_congest_idx = np.where(path_has_congest)[0]
            normal_idx = np.where(~path_has_congest)[0]

            global_limit = self.congestion_tracker.get_global_congested_limit(len(vehs))

            if high_congest_alloc > global_limit and len(high_congest_idx) > 0 and self.feedback_gain > 1e-6:
                excess = high_congest_alloc - global_limit
                reduction = self.feedback_gain * excess
                reduction = np.clip(reduction, 0.0, excess)

                if high_congest_alloc > 1e-6:
                    path_avg_risk = np.add.reduceat(congestion_risk[flat_edges], reduce_idx) / np.add.reduceat(
                        np.ones_like(congested_flags), reduce_idx)
                    congest_risk = path_avg_risk[high_congest_idx]
                    risk_sum = np.sum(congest_risk)
                    if risk_sum > 1e-6:
                        path_reduction = reduction * congest_risk / risk_sum
                        path_allocation[high_congest_idx] -= path_reduction
                        path_allocation[high_congest_idx] = np.maximum(path_allocation[high_congest_idx], 0.0)
                    else:
                        scale = (high_congest_alloc - reduction) / high_congest_alloc
                        path_allocation[high_congest_idx] *= scale

                if reduction > 1e-6:
                    if len(normal_idx) > 0:
                        normal_weights = path_weights[normal_idx]
                        normal_total_weight = np.sum(normal_weights)
                        if normal_total_weight > 1e-6:
                            path_allocation[normal_idx] += reduction * normal_weights / normal_total_weight
                        else:
                            path_allocation[normal_idx] += reduction / len(normal_idx)
                    else:
                        congest_risk_all = path_avg_risk[high_congest_idx]
                        inverse_risk = 1.0 / np.maximum(congest_risk_all, 1e-6)
                        inverse_risk_sum = np.sum(inverse_risk)
                        if inverse_risk_sum > 1e-6:
                            path_allocation[high_congest_idx] += reduction * inverse_risk / inverse_risk_sum
                        else:
                            high_weights = path_weights[high_congest_idx]
                            high_total_weight = np.sum(high_weights)
                            if high_total_weight > 1e-6:
                                path_allocation[high_congest_idx] += reduction * high_weights / high_total_weight
                            else:
                                path_allocation[high_congest_idx] += reduction / len(high_congest_idx)

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
            max_global_congested_ratio=self.congestion_tracker.max_global_congested_ratio,
            gamma_max=self.gamma_max
        )
        self._path_priority_cache.clear()
        self._path_edge_ids_cache.clear()
        self._path_edge_pos_cache.clear()
        self._path_edges_cache.clear()
        self._path_offsets_cache.clear()
        self._path_congest_cache.clear()
        self._path_reduce_idx_cache.clear()


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

    def record_step(self, pred_risk, real_occupancy):
        self.pred_risk_history.append(pred_risk.copy())
        self.real_occupancy_history.append(real_occupancy.copy())

    def _causal_moving_average(self, data):
        n = data.shape[0]
        w = self.window_size
        if n == 0:
            return data
        cumsum = np.cumsum(data, axis=0)
        result = np.zeros_like(data)
        for t in range(n):
            lo = t - w + 1
            if lo <= 0:
                result[t] = cumsum[t] / (t + 1)
            else:
                result[t] = (cumsum[t] - cumsum[lo - 1]) / w
        return result

    def calculate_metrics(self):
        pred_risks = np.array(self.pred_risk_history)
        real_occs = np.array(self.real_occupancy_history)
        real_smoothed = self._causal_moving_average(real_occs)

        lead_time_array = np.full(self.num_edges, np.nan, dtype=np.float64)
        miss_flags = np.zeros(self.num_edges, dtype=bool)
        late_flags = np.zeros(self.num_edges, dtype=bool)
        pred_exists_flags = np.zeros(self.num_edges, dtype=bool)
        real_exists_flags = np.zeros(self.num_edges, dtype=bool)
        true_positive_flags = np.zeros(self.num_edges, dtype=bool)

        for e_idx in range(self.num_edges):
            pred_series = pred_risks[:, e_idx]
            real_series = real_smoothed[:, e_idx]

            real_first_idx = np.where(real_series >= self.real_congestion_threshold)[0]
            pred_first_idx = np.where(pred_series >= self.business_threshold[e_idx])[0]

            has_pred = len(pred_first_idx) > 0
            has_real = len(real_first_idx) > 0
            pred_exists_flags[e_idx] = has_pred
            real_exists_flags[e_idx] = has_real

            if not has_pred or not has_real:
                miss_flags[e_idx] = True
                continue

            lead = real_first_idx[0] - pred_first_idx[0]
            if 0 <= lead <= LEAD_TIME_MAX_WINDOW:
                lead_time_array[e_idx] = lead
                true_positive_flags[e_idx] = True
            else:
                late_flags[e_idx] = True

        valid_leads = lead_time_array[~np.isnan(lead_time_array)]
        avg_lead_second = float(np.mean(valid_leads)) if valid_leads.size > 0 else 0.0
        median_lead_second = float(np.median(valid_leads)) if valid_leads.size > 0 else 0.0
        miss_rate = float(np.mean(miss_flags))
        late_rate = float(np.mean(late_flags))

        triggered_edge_count = int(np.sum(pred_exists_flags))
        congested_edge_count = int(np.sum(real_exists_flags))
        true_positive_count = int(np.sum(true_positive_flags))
        precision = (true_positive_count / triggered_edge_count) if triggered_edge_count > 0 else 0.0
        recall = (true_positive_count / congested_edge_count) if congested_edge_count > 0 else 0.0

        business_mask = pred_risks >= self.business_threshold
        business_trigger_ratio = np.sum(business_mask) / pred_risks.size

        real_cong_mask = real_smoothed >= self.real_congestion_threshold
        total_congestion_time = float(np.sum(real_cong_mask))

        recovery_times = []
        excluded_sample_count = 0
        for edge_idx in range(self.num_edges):
            edge_real = real_smoothed[:, edge_idx]
            peak_idx = np.argmax(edge_real)
            peak_val = edge_real[peak_idx]
            if peak_val < self.real_congestion_threshold:
                excluded_sample_count += 1
                continue
            follow_real = edge_real[peak_idx:]
            below_pos = np.where(follow_real < self.real_congestion_threshold)[0]
            if len(below_pos) > 0:
                recovery_times.append(below_pos[0])
            else:
                excluded_sample_count += 1
        avg_recovery_time = np.mean(recovery_times) if len(recovery_times) > 0 else 0.0

        return {
            "avg_lead_second": avg_lead_second,
            "median_lead_second": median_lead_second,
            "lead_time_distribution": lead_time_array,
            "miss_rate": miss_rate,
            "late_rate": late_rate,
            "precision": precision,
            "recall": recall,
            "triggered_edge_count": triggered_edge_count,
            "congested_edge_count": congested_edge_count,
            "business_trigger_ratio": business_trigger_ratio,
            "total_congestion_time": total_congestion_time,
            "avg_recovery_time": avg_recovery_time,
            "excluded_sample_count": excluded_sample_count
        }

    def calculate_lead_time(self, target_edge_idx):
        real_thresh = self.real_congestion_threshold
        pred_thresh = self.business_threshold[target_edge_idx]

        pred_array = np.array(self.pred_risk_history)[:, target_edge_idx]
        real_array = self._causal_moving_average(np.array(self.real_occupancy_history))[:, target_edge_idx]

        pred_times = np.where(pred_array >= pred_thresh)[0]
        real_times = np.where(real_array >= real_thresh)[0]

        if len(pred_times) == 0 or len(real_times) == 0:
            return np.nan
        lead = real_times[0] - pred_times[0]
        return lead if 0 <= lead <= LEAD_TIME_MAX_WINDOW else np.nan


def preload_trip_full(trip_file, edge_from_node, edge_to_node):
    try:
        tree = ET.parse(trip_file)
        root = tree.getroot()
        veh_info = {}
        junction_out_edges = defaultdict(list)
        for eid, n in edge_from_node.items():
            junction_out_edges[n].append(eid)

        for trip in root.findall("trip"):
            vid = trip.get("id")
            o_junc_str = trip.get("fromJunction")
            d_junc_str = trip.get("toJunction")
            if not vid or not o_junc_str or not d_junc_str:
                continue
            o_node = int(o_junc_str)
            d_node = int(d_junc_str)
            if o_node not in junction_out_edges or len(junction_out_edges[o_node]) == 0:
                continue
            from_eid = junction_out_edges[o_node][0]
            veh_info[vid] = (o_node, d_node, from_eid)
        return veh_info
    except Exception as e:
        print(f"[preload_trip_full] parse exception: {str(e)}")
        return {}


def build_adjacency_matrix(edges, edge_id_to_idx):
    num_edges = len(edges)
    from_node_to_indices = {}
    for j, e2 in enumerate(edges):
        from_node_to_indices.setdefault(e2.getFromNode().getID(), []).append(j)
    rows = []
    cols = []
    for i, e in enumerate(edges):
        dst_node = e.getToNode().getID()
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


def truncate_candidate_paths(od_candidate_paths, max_paths):
    if max_paths is None:
        return od_candidate_paths
    truncated = {}
    for k, v in od_candidate_paths.items():
        truncated[k] = v[:max_paths] if len(v) > max_paths else v
    return truncated


def build_sumo_cmd(net_file, trips_file, sim_end, seed, output_tripinfo=None):
    cmd = [
        SUMO_EXE,
        "-n", net_file,
        "-r", trips_file,
        "--begin", "0",
        "--end", str(sim_end),
        "--step-length", "1",
        "--junction-taz",
        "--no-step-log",
        "--xml-validation", "never",
        "--no-warnings",
        "--time-to-teleport", "300",
        "--seed", str(seed)
    ]
    if output_tripinfo is not None:
        cmd.extend(["--tripinfo-output", output_tripinfo])
    return cmd


def collect_calibration_occupancy(net_file, trips_file, edges, sim_duration, seed):
    sumo_cmd = build_sumo_cmd(net_file, trips_file, sim_duration, seed)
    traci.start(sumo_cmd)
    for e in edges:
        traci.edge.subscribe(e.getID(), [tc.LAST_STEP_OCCUPANCY])
    samples = []
    sim_step = 0
    while sim_step < sim_duration:
        traci.simulationStep()
        sim_step += 1
        if sim_step <= WARMUP_DURATION:
            continue
        subscription_results = traci.edge.getAllSubscriptionResults()
        occ = np.zeros(len(edges), dtype=np.float32)
        for i, e in enumerate(edges):
            result = subscription_results.get(e.getID())
            if result is None:
                continue
            occ[i] = min(result.get(tc.LAST_STEP_OCCUPANCY, 0.0) / 100.0, 1.0)
        samples.append(occ)
    traci.close()
    return np.array(samples, dtype=np.float32)


def compute_empirical_thresholds(calibration_samples, quantiles):
    flat = calibration_samples.reshape(-1)
    return [float(np.quantile(flat, q)) for q in quantiles]


def run_single_simulation(strategy_name, allocator, net, edges, edge_id_to_idx, edge_idx_to_id,
                          free_flow_speeds, edge_lengths, edge_from_node, veh_map_local,
                          trips_file, sim_duration, seed, output_tripinfo):
    threshold = allocator.congestion_threshold
    evaluator = CongestionPredictionEvaluator(
        num_edges=len(edges),
        business_threshold=allocator.congestion_threshold,
        top_ratio=0.1
    )

    max_total_steps = sim_duration + CLEARANCE_DURATION
    sumo_cmd = build_sumo_cmd(NET_FILE, trips_file, max_total_steps, seed, output_tripinfo)

    traci.start(sumo_cmd)
    for e in edges:
        traci.edge.subscribe(e.getID(), [tc.LAST_STEP_OCCUPANCY, tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
    sim_step = 0
    assigned_count = 0
    total_algo_time = 0.0
    path_edge_ids = allocator._path_edge_ids_cache
    path_edge_pos = allocator._path_edge_pos_cache

    halting_count_history = []
    teleport_start_total = 0
    teleport_end_total = 0
    outflow_saturation_flags = []
    last_real_occupancy = np.zeros(len(edges), dtype=np.float32)

    with tqdm(total=max_total_steps, desc=f"{strategy_name}", unit="step", ncols=100) as pbar:
        while sim_step < max_total_steps:
            traci.simulationStep()
            sim_step += 1
            pbar.update(1)

            subscription_results = traci.edge.getAllSubscriptionResults()
            real_occupancy = np.zeros(len(edges), dtype=np.float32)
            halting_total = 0
            for i, e in enumerate(edges):
                result = subscription_results.get(e.getID())
                if result is None:
                    continue
                occ = result.get(tc.LAST_STEP_OCCUPANCY, 0.0) / 100.0
                real_occupancy[i] = min(occ, 1.0)
                halting_total += result.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            halting_count_history.append(halting_total)

            if sim_step % 10 == 0:
                last_real_occupancy = real_occupancy.copy()
            else:
                real_occupancy = last_real_occupancy.copy()

            try:
                teleport_start_total += traci.simulation.getStartingTeleportNumber()
                teleport_end_total += traci.simulation.getEndingTeleportNumber()
            except:
                pass

            outflow_saturation_flags.append(bool(np.mean(real_occupancy >= 0.95) > 0.05))

            if sim_step > WARMUP_DURATION:
                t0 = time.perf_counter()
                pred_risk = allocator.calculate_congestion_risk()
                evaluator.record_step(pred_risk, real_occupancy)
                total_algo_time += time.perf_counter() - t0

            departed = traci.simulation.getDepartedIDList()
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
                    alloc_results = allocator.allocate_vehicles(vehicle_list, free_flow_speeds, edge_lengths)
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

            if sim_step >= sim_duration and traci.simulation.getMinExpectedNumber() == 0:
                break

    incomplete_vehicle_count = traci.simulation.getMinExpectedNumber()
    traci.close()
    metrics = evaluator.calculate_metrics()

    avg_real_occ = np.mean(evaluator.real_occupancy_history, axis=0)
    top_edges = np.argsort(avg_real_occ)[-5:]
    lead_times = [evaluator.calculate_lead_time(e) for e in top_edges]
    valid_lead_times = [t for t in lead_times if not np.isnan(t)]
    lead_time = np.mean(valid_lead_times) if len(valid_lead_times) > 0 else np.nan

    trip_durations = []
    tree = ET.parse(output_tripinfo)
    for trip in tree.getroot().findall('tripinfo'):
        duration = float(trip.get('duration', 0))
        trip_durations.append(duration)

    avg_trip_time = np.mean(trip_durations) if len(trip_durations) > 0 else 0
    network_metrics = {
        "avg_trip_time": avg_trip_time,
        "total_completed": len(trip_durations)
    }

    efficiency_metrics = {
        "total_algo_time": total_algo_time,
        "computation_latency_ms": total_algo_time / sim_duration * 1000,
        "throughput": sim_duration / total_algo_time if total_algo_time > 0 else 0,
        "computation_latency_per_edge_ms": (total_algo_time / sim_duration * 1000) / len(edges)
    }

    diagnostics = {
        "mean_halting_count": float(np.mean(halting_count_history)) if halting_count_history else 0.0,
        "max_halting_count": float(np.max(halting_count_history)) if halting_count_history else 0.0,
        "teleport_start_total": teleport_start_total,
        "teleport_end_total": teleport_end_total,
        "outflow_saturation_step_ratio": float(np.mean(outflow_saturation_flags)) if outflow_saturation_flags else 0.0,
        "incomplete_vehicle_count": int(incomplete_vehicle_count),
        "incomplete_ratio": incomplete_vehicle_count / max(len(veh_map_local), 1),
        "traffic_loading_model": TRAFFIC_LOADING_MODEL
    }

    return metrics, lead_time, assigned_count, network_metrics, efficiency_metrics, diagnostics


def run_sweep_experiment(sweep_grid, thresh, edgewise_threshold, capacity_array,
                          net, edges, edge_id_to_idx, edge_idx_to_id,
                          free_flow_speeds, edge_lengths, edge_from_node,
                          veh_map_local, trips_file, sim_duration, seed,
                          od_candidate_paths, base_output_dir="./sweep_results"):
    os.makedirs(base_output_dir, exist_ok=True)
    results = []

    for point_idx, point in enumerate(sweep_grid):
        kp = point.get("feedback_gain", 0.02)
        t_w = point.get("smooth_window", 60)
        alpha = point.get("time_decay_factor", 0.95)
        theta_scale = point.get("theta_scale", 1.0)
        gamma_max = point.get("gamma_max", GAMMA_MAX_PRIMARY)
        n_paths = point.get("num_candidate_paths", None)

        point_threshold = np.clip(edgewise_threshold * theta_scale, 0.0, 1.0).astype(np.float32)
        point_paths = truncate_candidate_paths(od_candidate_paths, n_paths)

        is_open_loop = kp <= 1e-6
        strategy_label = "Detection-only (open-loop, no actuation)" if is_open_loop else f"Kp_{kp}"
        strategy_name = f"{strategy_label}_Tw{t_w}_alpha{alpha}_gamma{gamma_max}_paths{n_paths}"
        print(f"\nrunning sweep point {point_idx}: {strategy_name}")

        allocator = EndogenousRouteAllocator(
            num_edges=len(edges),
            od_candidate_paths=point_paths,
            congestion_threshold=point_threshold,
            smooth_window=t_w,
            time_decay_factor=alpha,
            congestion_update_interval=1,
            use_d3stn_enhance=False,
            gamma_max=gamma_max,
            feedback_gain=kp
        )
        allocator.set_edge_capacity(
            capacity_array * allocator.congestion_tracker.time_window_size / 3600 * 0.08
        )
        allocator.precompute_path_data(free_flow_speeds, edge_lengths, edge_idx_to_id)

        output_tripinfo = os.path.join(
            base_output_dir,
            f"tripinfo_sweep_{point_idx}_thresh_{thresh:.4f}.xml"
        )

        metrics, lead_time, assigned, net_metrics, eff_metrics, diagnostics = run_single_simulation(
            strategy_name=strategy_name,
            allocator=allocator,
            net=net,
            edges=edges,
            edge_id_to_idx=edge_id_to_idx,
            edge_idx_to_id=edge_idx_to_id,
            free_flow_speeds=free_flow_speeds,
            edge_lengths=edge_lengths,
            edge_from_node=edge_from_node,
            veh_map_local=veh_map_local,
            trips_file=trips_file,
            sim_duration=sim_duration,
            seed=seed,
            output_tripinfo=output_tripinfo
        )

        results.append({
            "point": point,
            "strategy_name": strategy_name,
            "metrics": metrics,
            "lead_time": lead_time,
            "assigned": assigned,
            "network_metrics": net_metrics,
            "efficiency_metrics": eff_metrics,
            "diagnostics": diagnostics
        })

    return results


def run_kp_ablation_experiment(thresh, edgewise_threshold, capacity_array,
                               net, edges, edge_id_to_idx, edge_idx_to_id,
                               free_flow_speeds, edge_lengths, edge_from_node,
                               veh_map_local, trips_file, sim_duration, seed,
                               od_candidate_paths, base_output_dir="./kp_ablation_results",
                               gamma_max=GAMMA_MAX_PRIMARY):
    sweep_grid = [{"feedback_gain": kp, "gamma_max": gamma_max} for kp in KP_VALUES]

    sweep_results = run_sweep_experiment(
        sweep_grid=sweep_grid,
        thresh=thresh,
        edgewise_threshold=edgewise_threshold,
        capacity_array=capacity_array,
        net=net,
        edges=edges,
        edge_id_to_idx=edge_id_to_idx,
        edge_idx_to_id=edge_idx_to_id,
        free_flow_speeds=free_flow_speeds,
        edge_lengths=edge_lengths,
        edge_from_node=edge_from_node,
        veh_map_local=veh_map_local,
        trips_file=trips_file,
        sim_duration=sim_duration,
        seed=seed,
        od_candidate_paths=od_candidate_paths,
        base_output_dir=base_output_dir
    )

    kp_result_list = []
    for kp, res in zip(KP_VALUES, sweep_results):
        kp_result_list.append((
            res["metrics"], res["lead_time"], res["assigned"],
            res["network_metrics"], res["efficiency_metrics"], res["diagnostics"]
        ))
        print(
            f"  done {res['strategy_name']} | "
            f"avg_trip_time={res['network_metrics']['avg_trip_time']:.1f}s | "
            f"total_congestion_time={res['metrics']['total_congestion_time']:.0f} | "
            f"trigger_ratio={res['metrics']['business_trigger_ratio']*100:.2f}%"
        )

    return kp_result_list


def plot_bar_single_kp(thresh, kp_values, metrics_list, save_path="bar_kp_thresh.pdf"):
    metric_keys = [
        "avg_trip_time",
        "total_congestion_time",
        "avg_recovery_time",
        "avg_lead_second",
        "business_trigger_ratio",
        "computation_latency_ms"
    ]
    metric_labels = [
        "Avg Trip Time (s)",
        "Total Congestion Time (edge*s)",
        "Peak Recovery Time (s)",
        "Avg Lead Time (s)",
        "Diversion Trigger Ratio",
        "Computation Latency (CPU, ms/step)"
    ]

    x = np.arange(len(kp_values))
    width = 0.65
    base_name = save_path.rsplit('.', 1)[0]
    kp_labels = [f"Kp={k}" for k in kp_values]

    for key, label in zip(metric_keys, metric_labels):
        fig, ax = plt.subplots(figsize=(7, 5))
        values = [m.get(key, 0.0) for m in metrics_list]
        ax.bar(x, values, width, color=COLORS[:len(kp_values)], edgecolor='black', linewidth=0.8)

        ax.set_title(f"{label}\n(Threshold = {thresh:.4f})", fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(kp_labels, rotation=0, ha='center', fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{base_name}_{key}.pdf", format='pdf', bbox_inches='tight')
        plt.close()


def plot_trend_kp_all(thresholds, kp_values, all_metrics, save_path="kp_trend_all.pdf"):
    metric_keys = [
        "avg_trip_time",
        "total_congestion_time",
        "avg_recovery_time",
        "avg_lead_second",
        "business_trigger_ratio",
        "computation_latency_ms"
    ]
    metric_labels = [
        "Avg Trip Time (s)",
        "Total Congestion Time (edge*s)",
        "Peak Recovery Time (s)",
        "Avg Lead Time (s)",
        "Diversion Trigger Ratio",
        "Computation Latency (CPU, ms/step)"
    ]

    base_name = save_path.rsplit('.', 1)[0]
    kp_labels = [f"Kp={k}" for k in kp_values]

    for key, label in zip(metric_keys, metric_labels):
        fig, ax = plt.subplots(figsize=(7, 5))

        n_kp = len(kp_values)
        for k_idx in range(n_kp):
            values = [all_metrics[t_idx][k_idx].get(key, 0.0) for t_idx in range(len(thresholds))]
            ax.plot(thresholds, values, color=COLORS[k_idx], marker=MARKERS[k_idx],
                    linewidth=1.5, markersize=5, label=kp_labels[k_idx])

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


def calculate_threshold_sensitivity(x_values, trigger_ratios):
    slope, _, _, _, _ = linregress(x_values, trigger_ratios)
    return -slope


def build_kp_ablation_row(kp, gamma_max, result_tuple):
    metrics, lead_time, assigned, net_metrics, eff_metrics, diagnostics = result_tuple
    is_open_loop = kp <= 1e-6
    kp_label = "Free-flow weighted baseline (Kp=0.00)" if is_open_loop else f"Kp={kp:.2f}"
    diversion_mode = "Detection-only (open-loop, no actuation)" if is_open_loop else "Closed-loop actuation"

    return {
        "kp_label": kp_label,
        "gamma_max": gamma_max,
        "diversion_mode": diversion_mode,
        "avg_trip_time_s": net_metrics["avg_trip_time"],
        "total_congestion_time_edge_s": metrics["total_congestion_time"],
        "avg_recovery_time_s": metrics["avg_recovery_time"],
        "recovery_excluded_sample_count": metrics["excluded_sample_count"],
        "avg_lead_second": metrics["avg_lead_second"],
        "median_lead_second": metrics["median_lead_second"],
        "lead_time_miss_rate": metrics["miss_rate"],
        "lead_time_late_rate": metrics["late_rate"],
        "trigger_ratio": metrics["business_trigger_ratio"],
        "computation_latency_ms_per_step": eff_metrics["computation_latency_ms"],
        "assigned_count": assigned,
        "incomplete_vehicle_count": diagnostics["incomplete_vehicle_count"],
        "incomplete_ratio": diagnostics["incomplete_ratio"],
        "mean_halting_count": diagnostics["mean_halting_count"],
        "teleport_start_total": diagnostics["teleport_start_total"],
        "teleport_end_total": diagnostics["teleport_end_total"],
        "outflow_saturation_step_ratio": diagnostics["outflow_saturation_step_ratio"]
    }


def print_kp_ablation_table(thresh, primary_results, hard_saturation_results):
    rows = []
    for kp, res in zip(KP_VALUES, primary_results):
        rows.append(build_kp_ablation_row(kp, GAMMA_MAX_PRIMARY, res))
    for kp, res in zip(KP_VALUES, hard_saturation_results):
        rows.append(build_kp_ablation_row(kp, GAMMA_MAX_HARD_SATURATION, res))

    df = pandas.DataFrame(rows)

    print("\n" + "=" * 120)
    print(f"Table II: Kp Ablation Results | Traffic Loading Model = {TRAFFIC_LOADING_MODEL} | Threshold = {thresh:.4f}")
    print("=" * 120)

    primary_df = df[df["gamma_max"] == GAMMA_MAX_PRIMARY].set_index("kp_label")
    hard_df = df[df["gamma_max"] == GAMMA_MAX_HARD_SATURATION].set_index("kp_label")

    print(f"\nPrimary Sweep (gamma_max = {GAMMA_MAX_PRIMARY})")
    print(primary_df.to_string())

    print(f"\nHard-Saturation Comparison (gamma_max={GAMMA_MAX_HARD_SATURATION})")
    print(hard_df.to_string())

    print("\nLead Time Distribution Summary (per-edge, seconds, NaN = no valid detection)")
    for kp, res in zip(KP_VALUES, primary_results):
        metrics = res[0]
        dist = metrics["lead_time_distribution"]
        valid = dist[~np.isnan(dist)]
        if valid.size > 0:
            p25, p50, p75 = np.percentile(valid, [25, 50, 75])
        else:
            p25, p50, p75 = np.nan, np.nan, np.nan
        print(
            f"  Kp={kp:.2f} gamma_max={GAMMA_MAX_PRIMARY} | "
            f"valid_samples={valid.size}/{dist.size} | "
            f"p25={p25:.1f} p50={p50:.1f} p75={p75:.1f}"
        )

    print("=" * 120)
    return df


if __name__ == "__main__":
    NETWORK_NAME = "Philadelphia"
    NETWORK_SCALE_RATIO = 0.2
    TARGET_TRIP_COUNT = 1_000_000
    NET_FILE = f"{NETWORK_NAME}_cropped_{NETWORK_SCALE_RATIO}.net.xml"
    CAPACITY_CSV = f"edge_capacity_{NETWORK_NAME.lower()}.csv"
    CANDIDATE_PKL = f"od_candidate_paths_{NETWORK_NAME.lower()}.pkl"
    TRIPS_FILE = f"{NETWORK_NAME}_filtered_{NETWORK_SCALE_RATIO}_1800_{TARGET_TRIP_COUNT}_v4.trips.xml"
    SIM_DURATION = 1800
    SEED = TEST_SEEDS[0]

    net = sumolib.net.readNet(NET_FILE)
    all_edges = net.getEdges()

    if NETWORK_SCALE_RATIO < 1.0:
        bbox = net.getBBoxXY()
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        x_cut = x_min + (x_max - x_min) * np.sqrt(NETWORK_SCALE_RATIO)
        y_cut = y_min + (y_max - y_min) * np.sqrt(NETWORK_SCALE_RATIO)
        edges = [e for e in all_edges if e.getFromNode().getCoord()[0] <= x_cut and e.getFromNode().getCoord()[1] <= y_cut]
    else:
        edges = all_edges

    edge_id_to_idx = {e.getID(): i for i, e in enumerate(edges)}
    edge_idx_to_id = {i: e.getID() for i, e in enumerate(edges)}
    num_edges = len(edges)

    full_edge_id_to_idx_for_pickle = {e.getID(): i for i, e in enumerate(all_edges)}

    free_flow_speeds = np.array([e.getSpeed() for e in edges], dtype=np.float32)
    edge_lengths = np.array([e.getLength() for e in edges], dtype=np.float32)
    edge_from_node = {e.getID(): int(e.getFromNode().getID()) for e in edges}

    veh_full_map = preload_trip_full(TRIPS_FILE, edge_from_node, {})
    _veh_pre_ids = []
    _veh_pre_rows = []
    for vid, (o, d, from_eid) in veh_full_map.items():
        if from_eid in edge_id_to_idx:
            _veh_pre_ids.append(vid)
            _veh_pre_rows.append((o, d, edge_id_to_idx[from_eid]))
    veh_pre_data = np.array(_veh_pre_rows, dtype=VEH_PRE_MAP_DTYPE)
    veh_pre_index = {vid: i for i, vid in enumerate(_veh_pre_ids)}
    veh_pre_map = VehPreMap(veh_pre_index, veh_pre_data)
    print(f"preloaded {len(veh_pre_map)} trips")

    if NETWORK_SCALE_RATIO < 1.0:
        import re
        filtered_trips = f"{NETWORK_NAME}_filtered_{NETWORK_SCALE_RATIO}.trips.xml"
        valid_vids = set(veh_pre_map._index.keys())
        id_pattern = re.compile(r'id="([^"]+)"')
        with open(TRIPS_FILE, 'r', encoding='utf-8') as fin, open(filtered_trips, 'w', encoding='utf-8') as fout:
            for line in fin:
                if '<trip ' in line:
                    match = id_pattern.search(line)
                    if match and match.group(1) in valid_vids:
                        fout.write(line)
                else:
                    fout.write(line)
        TRIPS_FILE = filtered_trips
        print(f"filtered trips file: {filtered_trips}")


    cap_df = pandas.read_csv(CAPACITY_CSV, index_col=0)
    capacity_array = np.zeros(num_edges, dtype=np.float32)
    for edge_id, row in cap_df.iterrows():
        if edge_id in edge_id_to_idx:
            capacity_array[edge_id_to_idx[edge_id]] = row["capacity_veh_per_hour"]

    with open(CANDIDATE_PKL, "rb") as f:
        data = pickle.load(f)
    _raw_od_candidate_paths = data["od_candidate_paths"]
    pickle_edge_id_to_idx = data.get("edge_id_to_idx", {})
    pickle_idx_to_edge_id = {v: k for k, v in pickle_edge_id_to_idx.items()}

    od_candidate_paths = {}
    for od_key, paths in _raw_od_candidate_paths.items():
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
    print(f"valid OD pairs: {len(od_candidate_paths)}, total paths: {sum(len(p) for p in od_candidate_paths.values())}")

    print("\nrunning calibration pass (CALIBRATION_SEEDS)")
    calibration_arrays = [
        collect_calibration_occupancy(NET_FILE, TRIPS_FILE, edges, SIM_DURATION, s)
        for s in CALIBRATION_SEEDS
    ]
    calibration_occupancy = np.concatenate(calibration_arrays, axis=0)

    print("running validation pass (VALIDATION_SEEDS)")
    validation_arrays = [
        collect_calibration_occupancy(NET_FILE, TRIPS_FILE, edges, SIM_DURATION, s)
        for s in VALIDATION_SEEDS
    ]
    validation_occupancy = np.concatenate(validation_arrays, axis=0)

    threshold_quantiles = [0.8, 0.9, 0.95, 0.99]
    threshold_list = compute_empirical_thresholds(calibration_occupancy, threshold_quantiles)
    validation_threshold_list = compute_empirical_thresholds(validation_occupancy, threshold_quantiles)

    print("\ncalibration vs validation empirical threshold check (seed-role separation)")
    for q, cal_t, val_t in zip(threshold_quantiles, threshold_list, validation_threshold_list):
        print(f"  quantile={q:.2f} calibration_threshold={cal_t:.5f} validation_threshold={val_t:.5f}")

    emp_min = float(np.quantile(calibration_occupancy.reshape(-1), 0.01))
    emp_max = float(np.quantile(calibration_occupancy.reshape(-1), 0.99))

    COL_WIDTH = 18

    def pct(v):
        return f"{v * 100:.2f}%"

    all_threshold_kp_results = []
    all_tables = []
    edgewise_threshold_list = []

    for thresh in threshold_list:
        print(f"\n\n{'=' * 100}")
        print(f"threshold group start: congestion_threshold = {thresh:.4f}")
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
        edgewise_threshold = np.clip(
            edgewise_threshold,
            a_min=emp_min * 0.5,
            a_max=emp_max * 1.2
        ).astype(np.float32)

        edgewise_threshold_list.append(edgewise_threshold)

        kp_result_list_primary = run_kp_ablation_experiment(
            thresh=thresh,
            edgewise_threshold=edgewise_threshold,
            capacity_array=capacity_array,
            net=net,
            edges=edges,
            edge_id_to_idx=edge_id_to_idx,
            edge_idx_to_id=edge_idx_to_id,
            free_flow_speeds=free_flow_speeds,
            edge_lengths=edge_lengths,
            edge_from_node=edge_from_node,
            veh_map_local=veh_pre_map,
            trips_file=TRIPS_FILE,
            sim_duration=SIM_DURATION,
            seed=SEED,
            od_candidate_paths=od_candidate_paths,
            base_output_dir="./kp_ablation_results/gamma_1p0",
            gamma_max=GAMMA_MAX_PRIMARY
        )

        kp_result_list_hard = run_kp_ablation_experiment(
            thresh=thresh,
            edgewise_threshold=edgewise_threshold,
            capacity_array=capacity_array,
            net=net,
            edges=edges,
            edge_id_to_idx=edge_id_to_idx,
            edge_idx_to_id=edge_idx_to_id,
            free_flow_speeds=free_flow_speeds,
            edge_lengths=edge_lengths,
            edge_from_node=edge_from_node,
            veh_map_local=veh_pre_map,
            trips_file=TRIPS_FILE,
            sim_duration=SIM_DURATION,
            seed=SEED,
            od_candidate_paths=od_candidate_paths,
            base_output_dir="./kp_ablation_results/gamma_0p4",
            gamma_max=GAMMA_MAX_HARD_SATURATION
        )

        table_df = print_kp_ablation_table(thresh, kp_result_list_primary, kp_result_list_hard)
        all_tables.append(table_df)

        thresh_merged = []
        for res in kp_result_list_primary:
            thresh_merged.append({**res[0], **res[3], **res[4]})
        all_threshold_kp_results.append(thresh_merged)

        if 'plot_bar_single_kp' in globals():
            plot_bar_single_kp(thresh, KP_VALUES, thresh_merged,
                               save_path=f"bar_kp_thresh_{thresh:.4f}.pdf")

    if 'plot_trend_kp_all' in globals():
        plot_trend_kp_all(threshold_list, KP_VALUES, all_threshold_kp_results,
                          save_path="kp_metric_trend_all.pdf")

    print("\n" + "=" * 100)
    print("Kp sensitivity analysis (trigger ratio rate of change vs Kp, primary sweep gamma_max=1.0)")
    print("=" * 100)
    for t_idx, thresh in enumerate(threshold_list):
        trigger_ratios = [
            all_threshold_kp_results[t_idx][k_idx]['business_trigger_ratio']
            for k_idx in range(len(KP_VALUES))
        ]
        sensitivity = calculate_threshold_sensitivity(KP_VALUES, trigger_ratios)
        print(f"threshold {thresh:.4f}: sensitivity = {sensitivity:.4f} per unit Kp")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("Multi-Axis Sweep Demonstration (Kp x gamma_max)")
    print("=" * 100)
    demo_grid = [
        {"feedback_gain": kp, "gamma_max": g}
        for kp in [KP_VALUES[0], KP_VALUES[-1]]
        for g in [GAMMA_MAX_PRIMARY, GAMMA_MAX_HARD_SATURATION]
    ]
    demo_results = run_sweep_experiment(
        sweep_grid=demo_grid,
        thresh=threshold_list[0],
        edgewise_threshold=edgewise_threshold_list[0],
        capacity_array=capacity_array,
        net=net,
        edges=edges,
        edge_id_to_idx=edge_id_to_idx,
        edge_idx_to_id=edge_idx_to_id,
        free_flow_speeds=free_flow_speeds,
        edge_lengths=edge_lengths,
        edge_from_node=edge_from_node,
        veh_map_local=veh_pre_map,
        trips_file=TRIPS_FILE,
        sim_duration=SIM_DURATION,
        seed=SEED,
        od_candidate_paths=od_candidate_paths,
        base_output_dir="./kp_ablation_results/multi_axis_demo"
    )
    demo_rows = []
    for point, res in zip(demo_grid, demo_results):
        metrics = res["metrics"]
        eff = res["efficiency_metrics"]
        demo_rows.append({
            "kp": point["feedback_gain"],
            "gamma_max": point["gamma_max"],
            "trigger_ratio": metrics["business_trigger_ratio"],
            "computation_latency_ms_per_step": eff["computation_latency_ms"]
        })
    demo_df = pandas.DataFrame(demo_rows)
    print(demo_df.to_string())
    print("=" * 100)
