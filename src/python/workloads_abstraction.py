# event_abstraction.py

from __future__ import annotations

import math
from typing import Dict, List


def total_wcet(workload: dict) -> int:
    return int(sum(workload["wcets"]))


def node_resource_demand(workload: dict) -> Dict[int, int]:
    """
    将一个 workload 的执行需求按资源类型聚合。
    例如:
      wcets=[1,2], types=[CPU,GPU]
      -> {CPU: 1, GPU: 2}
    """
    demand = {}
    for wcet, proc_type in zip(workload["wcets"], workload["types"]):
        proc_type = int(proc_type)
        wcet = int(wcet)
        demand[proc_type] = demand.get(proc_type, 0) + wcet
    return demand


def max_sliding_sum(values: List[int], window_bins: int) -> int:
    """
    计算 trace 上长度为 window_bins 的滑窗最大和。
    """
    if not values:
        return 0
    if window_bins <= 1:
        return max(values)

    if window_bins >= len(values):
        return sum(values)

    current = sum(values[:window_bins])
    best = current
    for i in range(window_bins, len(values)):
        current += values[i] - values[i - window_bins]
        if current > best:
            best = current
    return best


def percentile_sliding_sum(values: List[int], window_bins: int, q: float) -> int:
    """
    计算 trace 上长度为 window_bins 的滑窗和的 q 分位数。
    """
    if not values:
        return 0
    if window_bins <= 1:
        samples = list(values)
    elif window_bins >= len(values):
        samples = [sum(values)]
    else:
        samples = []
        current = sum(values[:window_bins])
        samples.append(current)
        for i in range(window_bins, len(values)):
            current += values[i] - values[i - window_bins]
            samples.append(current)

    samples.sort()
    idx = int(math.ceil(q * len(samples))) - 1
    idx = max(0, min(idx, len(samples) - 1))
    return samples[idx]


def expected_arrivals_in_window(event_workload: dict, window_size: int) -> float:
    """
    计算长度为 window_size 的时间窗内的期望到达次数。
    支持:
      - bernoulli
      - poisson
      - trace
    """
    model = event_workload["arrival_model"]
    model_type = model["type"]
    base_window = int(model["base_window"])

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    num_base_windows = window_size / base_window

    if model_type == "bernoulli":
        prob = float(model["prob"])
        return num_base_windows * prob

    if model_type == "poisson":
        rate = float(model["rate_per_window"])
        return num_base_windows * rate

    if model_type == "trace":
        trace = event_workload.get("trace_counts", [])
        if not trace:
            return 0.0
        return float(sum(trace)) / float(len(trace)) * num_base_windows

    raise ValueError(f"Unsupported arrival model type: {model_type}")


def percentile_arrivals_in_window(event_workload: dict, window_size: int, q: float) -> int:
    """
    估计长度为 window_size 的时间窗内的 q 分位数到达次数。
    支持:
      - bernoulli: 正态近似
      - poisson:   正态近似
      - trace:     滑窗统计
    """
    model = event_workload["arrival_model"]
    model_type = model["type"]
    base_window = int(model["base_window"])
    n = max(1, int(round(window_size / base_window)))

    if model_type == "trace":
        trace = event_workload.get("trace_counts", [])
        return max(1, percentile_sliding_sum(trace, n, q))

    z_table = {
        0.90: 1.282,
        0.95: 1.645,
        0.975: 1.960,
        0.99: 2.326,
        0.995: 2.576,
        0.999: 3.090,
    }
    if q not in z_table:
        raise ValueError(f"Unsupported percentile {q}. Supported: {list(z_table.keys())}")
    z = z_table[q]

    if model_type == "bernoulli":
        p = float(model["prob"])
        mean = n * p
        var = n * p * (1.0 - p)
        approx = mean + z * math.sqrt(var)
        return max(1, math.ceil(approx))

    if model_type == "poisson":
        lam = n * float(model["rate_per_window"])
        approx = lam + z * math.sqrt(lam)
        return max(1, math.ceil(approx))

    raise ValueError(f"Unsupported arrival model type: {model_type}")


def peak_arrivals_in_window(event_workload: dict, window_size: int, peak_cfg: dict | None = None) -> int:
    """
    估计长度为 window_size 的时间窗内的峰值到达次数。
    优先级:
      1) workload["peak_arrivals_per_window"]
      2) trace_counts 滑窗最大值
      3) multiplier * mean
    """
    if "peak_arrivals_per_window" in event_workload:
        return max(1, int(event_workload["peak_arrivals_per_window"]))

    model = event_workload["arrival_model"]
    base_window = int(model["base_window"])
    bins = max(1, int(round(window_size / base_window)))

    if model["type"] == "trace":
        trace = event_workload.get("trace_counts", [])
        return max(1, max_sliding_sum(trace, bins))

    multiplier = 3.0
    if peak_cfg is not None:
        multiplier = float(peak_cfg.get("multiplier", 3.0))

    mean = expected_arrivals_in_window(event_workload, window_size)
    return max(1, math.ceil(multiplier * mean))


def build_single_node_surrogate(
    name: str,
    period: int,
    wcet: int,
    proc_type: int,
    source_events: List[str],
    abstraction_mode: str,
) -> dict:
    """
    将等效后的 reservation 表达为一个单节点任务。
    """
    return {
        "name": name,
        "category": "surrogate_event_reservation",
        "period": int(period),
        "wcets": [int(max(1, wcet))],
        "types": [int(proc_type)],
        "dependency": [],
        "source_events": list(source_events),
        "abstraction_mode": abstraction_mode,
    }


def abstract_event_workloads(
    event_workloads: List[dict],
    reservation_period: int,
    mode: str = "mean",
    percentile_q: float = 0.95,
    peak_cfg: dict | None = None,
    split_by_resource: bool = True,
) -> List[dict]:
    """
    将真实 event-driven workloads 等效为 surrogate reservation tasks。

    参数:
      reservation_period:
        surrogate 周期
      mode:
        "mean" / "percentile" / "peak"
      split_by_resource:
        True -> 按资源类型分别生成 surrogate
        False -> 聚合成一个 UNKNOWN 类型 surrogate
    """
    if reservation_period <= 0:
        raise ValueError("reservation_period must be positive")

    if not event_workloads:
        return []

    arrivals_per_event = {}
    for w in event_workloads:
        if mode == "mean":
            arrivals_per_event[w["name"]] = expected_arrivals_in_window(w, reservation_period)
        elif mode == "percentile":
            arrivals_per_event[w["name"]] = percentile_arrivals_in_window(w, reservation_period, percentile_q)
        elif mode == "peak":
            arrivals_per_event[w["name"]] = peak_arrivals_in_window(w, reservation_period, peak_cfg)
        else:
            raise ValueError(f"Unsupported abstraction mode: {mode}")

    # 聚合不同 event workload 的资源需求
    agg_budget = {}
    for w in event_workloads:
        multiplier = arrivals_per_event[w["name"]]
        per_event_demand = node_resource_demand(w)
        for proc_type, demand in per_event_demand.items():
            agg_budget[proc_type] = agg_budget.get(proc_type, 0.0) + multiplier * demand

    source_events = [w["name"] for w in event_workloads]

    if split_by_resource:
        surrogates = []
        for proc_type, budget in sorted(agg_budget.items(), key=lambda x: x[0]):
            name = f"event_reservation_p{proc_type}_{mode}"
            if mode == "percentile":
                name += f"_q{int(percentile_q * 100)}"
            surrogates.append(
                build_single_node_surrogate(
                    name=name,
                    period=reservation_period,
                    wcet=math.ceil(budget),
                    proc_type=proc_type,
                    source_events=source_events,
                    abstraction_mode=mode,
                )
            )
        return surrogates

    total_budget = math.ceil(sum(agg_budget.values()))
    return [
        {
            "name": f"event_reservation_all_{mode}",
            "category": "surrogate_event_reservation",
            "period": int(reservation_period),
            "wcets": [int(max(1, total_budget))],
            "types": [9],  # UNKNOWN
            "dependency": [],
            "source_events": source_events,
            "abstraction_mode": mode,
        }
    ]