import os
import time
import numpy as np

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import njit
from tqdm.auto import tqdm


@dataclass
class WMSResult:
    reward: float
    joint_mi: float
    individual_mi_sum: float
    wms: float


# =====================================================
# Numba: Entropy from sorted integer labels
# =====================================================

@njit(nogil=True)
def _entropy_sorted(labels):
    n = len(labels)

    if n == 0:
        return 0.0

    entropy = 0.0
    count = 1

    for i in range(1, n):
        if labels[i] == labels[i - 1]:
            count += 1
        else:
            p = count / n
            entropy -= p * np.log2(p)
            count = 1

    p = count / n
    entropy -= p * np.log2(p)

    return entropy


# =====================================================
# Numba: MI between binary source and categorical target
# =====================================================

@njit(nogil=True)
def _binary_source_mi(x, y_labels, num_y_states):
    """
    Exact empirical MI between a binary variable X
    and a categorical target Y.

    x: (num_samples,) binary uint8
    y_labels: (num_samples,) integer state IDs
    """

    n = len(x)

    if n == 0:
        return 0.0

    counts = np.zeros((2, num_y_states), dtype=np.int64)

    for t in range(n):
        counts[x[t], y_labels[t]] += 1

    count_x = np.zeros(2, dtype=np.int64)
    count_y = np.zeros(num_y_states, dtype=np.int64)

    for a in range(2):
        for b in range(num_y_states):
            count_x[a] += counts[a, b]
            count_y[b] += counts[a, b]

    mi = 0.0

    for a in range(2):
        for b in range(num_y_states):
            c = counts[a, b]

            if c > 0:
                mi += (
                    (c / n)
                    * np.log2(
                        (c * n) /
                        (count_x[a] * count_y[b])
                    )
                )

    return max(0.0, mi)


# =====================================================
# Numba: MI from precomputed categorical labels
# =====================================================

@njit(nogil=True)
def _categorical_mi(x_labels, y_labels):
    """
    Empirical MI using:
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """

    n = len(x_labels)

    x_sorted = np.sort(x_labels.copy())
    y_sorted = np.sort(y_labels.copy())

    hx = _entropy_sorted(x_sorted)
    hy = _entropy_sorted(y_sorted)

    # Count joint states by sorting lexicographically.
    # np.lexsort support may vary across Numba versions,
    # so use an integer pair encoding.
    num_y = np.max(y_labels) + 1

    joint = (
        x_labels.astype(np.int64) * num_y
        + y_labels.astype(np.int64)
    )

    joint_sorted = np.sort(joint)

    hxy = _entropy_sorted(joint_sorted)

    return max(0.0, hx + hy - hxy)


# =====================================================
# Convert binary rows to categorical state IDs
# =====================================================

def binary_rows_to_labels(data: np.ndarray) -> np.ndarray:
    """
    Assign a unique integer ID to each observed binary row.
    No exponential state-space allocation.
    """

    data = np.ascontiguousarray(data, dtype=np.uint8)

    packed = np.packbits(data, axis=1)

    row_dtype = np.dtype(
        (np.void, packed.shape[1])
    )

    packed_rows = packed.view(row_dtype).ravel()

    _, inverse = np.unique(
        packed_rows,
        return_inverse=True
    )

    return inverse.astype(np.int64)


# =====================================================
# Thread worker
# =====================================================

def _compute_individual_mi(
    i: int,
    data: np.ndarray,
    y_labels: np.ndarray,
    num_y_states: int
):
    x_i = np.ascontiguousarray(data[:-1, i])

    mi = _binary_source_mi(
        x_i,
        y_labels,
        num_y_states
    )

    return i, float(mi)


# =====================================================
# WMS Calculator
# =====================================================

class WMSCalculator:

    def __init__(
        self,
        T: int | None = None,
        eps: float = 1e-12,
        show_progress: bool = True
    ):

        if hasattr(os, "process_cpu_count"):
            cpu_count = os.process_cpu_count()
        elif hasattr(os, "sched_getaffinity"):
            cpu_count = len(os.sched_getaffinity(0))
        else:
            cpu_count = os.cpu_count()

        cpu_count = cpu_count or 1

        max_threads = max(1, cpu_count // 2)

        if T is not None and T < 1:
            raise ValueError("T must be >= 1.")

        self.T = (
            min(T, max_threads)
            if T is not None
            else max_threads
        )

        self.eps = eps
        self.show_progress = show_progress

        print(f"Available logical CPUs: {cpu_count}")
        print(f"Maximum allowed threads: {max_threads}")
        print(f"Selected threads: {self.T}")

    def calculate(
        self,
        data: np.ndarray
    ) -> WMSResult:

        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(
                "data must have shape (num_samples, N)."
            )

        if data.shape[0] < 2:
            raise ValueError(
                "At least two time steps are required."
            )

        if not np.all((data == 0) | (data == 1)):
            raise ValueError(
                "data must contain only 0 and 1."
            )

        data = np.ascontiguousarray(
            data,
            dtype=np.uint8
        )

        num_samples, N = data.shape

        # =====================================
        # Step 1: Encode observed states
        # =====================================

        if self.show_progress:
            print("Encoding binary states...")

        x_labels = binary_rows_to_labels(
            data[:-1]
        )

        y_labels = binary_rows_to_labels(
            data[1:]
        )

        num_y_states = int(
            np.max(y_labels) + 1
        )

        # =====================================
        # Step 2: Joint MI
        # =====================================

        if self.show_progress:
            print("Computing joint MI...")

        A = _categorical_mi(
            x_labels,
            y_labels
        )

        # =====================================
        # Step 3: Individual MI in threads
        # =====================================

        individual_mis = np.zeros(
            N,
            dtype=np.float64
        )

        with ThreadPoolExecutor(
            max_workers=min(self.T, N)
        ) as executor:

            futures = [
                executor.submit(
                    _compute_individual_mi,
                    i,
                    data,
                    y_labels,
                    num_y_states
                )
                for i in range(N)
            ]

            with tqdm(
                total=N,
                desc="Computing Individual MI",
                unit="dimension",
                disable=not self.show_progress
            ) as pbar:

                for future in as_completed(futures):

                    i, mi = future.result()

                    individual_mis[i] = mi

                    pbar.update(1)

        B = float(np.sum(individual_mis))

        # =====================================
        # Step 4: WMS and reward
        # =====================================

        wms = float(A - B)

        reward = (
            A / (A + B)
            if A + B > self.eps
            else 0.5
        )

        return WMSResult(
            reward=float(reward),
            joint_mi=float(A),
            individual_mi_sum=B,
            wms=wms
        )


# =====================================================
# Example
# =====================================================

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    num_samples = 10000
    N = 4096
    T = 16

    data = rng.integers(
        0,
        2,
        size=(num_samples, N),
        dtype=np.uint8
    )

    calculator = WMSCalculator(
        T=T,
        show_progress=True
    )

    start = time.perf_counter()

    result = calculator.calculate(data)

    elapsed = time.perf_counter() - start

    print("\n========== WMS RESULTS ==========")
    print(f"Threads:       {calculator.T}")
    print(f"Joint MI:      {result.joint_mi:.6f} bits")
    print(f"Individual MI: {result.individual_mi_sum:.6f} bits")
    print(f"WMS:           {result.wms:.6f} bits")
    print(f"WMS Reward:    {result.reward:.6f}")
    print(f"Elapsed Time:  {elapsed:.4f} seconds")