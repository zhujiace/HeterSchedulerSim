# workloads.py

from copy import deepcopy

CPU = 0
DATACOPY = 3
GPU = 7
FPGA = 8
UNKNOWN = 9


WORKLOADS = {
    "always_on": {
        "vit_small": {
            "name": "vit_small",
            "category": "always_on",
            "period": 80,
            "wcets": [6, 2, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 2],
            "types": [CPU, GPU, CPU, GPU, GPU, GPU, GPU, GPU, CPU, GPU, CPU, GPU, CPU],
            "dependency": [
                0, 1,
                1, 2,
                2, 3,
                2, 4,
                2, 5,
                3, 6,
                4, 6,
                5, 7,
                6, 7,
                7, 8,
                8, 9,
                9, 10,
                10, 11,
                11, 12,
            ],
        },
        "vit_base": {
            "name": "vit_base",
            "category": "always_on",
            "period": 140,
            "wcets": [19, 5, 2, 1, 1, 1, 4, 8, 3, 2, 9, 1, 2],
            "types": [CPU, GPU, CPU, GPU, GPU, GPU, GPU, GPU, CPU, GPU, CPU, GPU, CPU],
            "dependency": [
                0, 1,
                1, 2,
                2, 3,
                2, 4,
                2, 5,
                3, 6,
                4, 6,
                5, 7,
                6, 7,
                7, 8,
                8, 9,
                9, 10,
                10, 11,
                11, 12,
            ],
        },
        "mobilevit_mixed": {
            "name": "mobilevit_mixed",
            "category": "always_on",
            "period": 100,
            "wcets": [3, 2, 4, 2, 5],
            "types": [CPU, GPU, GPU, CPU, GPU],
            "dependency": [
                0, 1,
                1, 2,
                1, 3,
                2, 4,
                3, 4,
            ],
        },
    },

    "event_driven": {
        # Bernoulli:
        # 每个 base_window 内，最多到达 1 次，概率为 prob
        "obstacle_alert_gpu": {
            "name": "obstacle_alert_gpu",
            "category": "event_driven",
            "wcets": [2, 3, 2],
            "types": [GPU, GPU, GPU],
            "dependency": [0, 1, 1, 2],
            "latency_target": 20,
            "arrival_model": {
                "type": "bernoulli",
                "prob": 0.12,
                "base_window": 10,
            },
        },

        # Poisson:
        # 每个 base_window 内，平均到达 rate_per_window 次，可出现 0/1/2/... 次
        "sign_detect_cpu_gpu": {
            "name": "sign_detect_cpu_gpu",
            "category": "event_driven",
            "wcets": [1, 2],
            "types": [CPU, GPU],
            "dependency": [0, 1],
            "latency_target": 30,
            "arrival_model": {
                "type": "poisson",
                "rate_per_window": 0.20,
                "base_window": 10,
            },
        },

        # Trace:
        # trace_counts[k] 表示第 k 个 base_window 内到达了多少次
        "pedestrian_popup_gpu": {
            "name": "pedestrian_popup_gpu",
            "category": "event_driven",
            "wcets": [1, 1, 2],
            "types": [GPU, GPU, GPU],
            "dependency": [0, 1, 1, 2],
            "latency_target": 25,
            "arrival_model": {
                "type": "trace",
                "base_window": 10,
            },
            "trace_counts": [0, 1, 0, 2, 1, 0, 0, 3, 1, 0, 1, 0, 2, 0, 0, 1],
        },
    }
}


def get_always_on_workload(name: str) -> dict:
    if name not in WORKLOADS["always_on"]:
        raise ValueError(f"Unknown always-on workload: {name}")
    return deepcopy(WORKLOADS["always_on"][name])


def get_event_workload(name: str) -> dict:
    if name not in WORKLOADS["event_driven"]:
        raise ValueError(f"Unknown event-driven workload: {name}")
    return deepcopy(WORKLOADS["event_driven"][name])


# CPU = 0
# DATACOPY = 3
# GPU = 7
# FPGA = 8
# UNKNOWN = 9


# WORKLOADS = {
#     "vit_small": {
#         "period": 80,
#         "wcets": [6, 2, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 2],
#         "types": [CPU, GPU, CPU, GPU, GPU, GPU, GPU, GPU, CPU, GPU, CPU, GPU, CPU],
#         "dependency": [0,1, 1,2, 2,3, 2,4, 2,5, 3,6, 4,6, 5,7, 6,7, 7,8, 8,9, 9,10, 10,11, 11,12,],
#     },

#     "vit_base": {
#         "period": 140,
#         "wcets": [19, 5, 2, 1, 1, 1, 4, 8, 3, 2, 9, 1, 2],
#         "types": [CPU, GPU, CPU, GPU, GPU, GPU, GPU, GPU, CPU, GPU, CPU, GPU, CPU],
#         "dependency": [0,1, 1,2, 2,3, 2,4, 2,5, 3,6, 4,6, 5,7, 6,7, 7,8, 8,9, 9,10, 10,11, 11,12,],
#     },

#     "vit_large": {
#         "period": 260,
#         "wcets": [56, 8, 4, 2, 2, 1, 12, 17, 4, 3, 11, 3, 3],
#         "types": [CPU, GPU, CPU, GPU, GPU, GPU, GPU, GPU, CPU, GPU, CPU, GPU, CPU],
#         "dependency": [0,1, 1,2, 2,3, 2,4, 2,5, 3,6, 4,6, 5,7, 6,7, 7,8, 8,9, 9,10, 10,11, 11,12,],
#     },
# }