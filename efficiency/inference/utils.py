import time
from typing import List

import numpy as np
import torch

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

def cpu_time() -> float:
    torch.cpu.synchronize()
    return time.perf_counter()


def stable_mean(arr: List[float]) -> float:
    if len(arr) < 4:
        return np.mean(arr)
    size = int(len(arr) * 0.25)
    return np.mean(sorted(arr)[size:-size])