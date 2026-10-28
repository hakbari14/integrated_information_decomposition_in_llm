
import time
import numpy as np
import os

from collections import Counter
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from multiprocessing import shared_memory
from dataclasses import dataclass
from tqdm.auto import tqdm


@dataclass
class WMSResult:
    reward: float
    joint_mi: float
    individual_mi_sum: float
    wms: float


def mutual_information(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Empirical discrete mutual information in bits.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 1:
        x = x[:, None]

    if y.ndim == 1:
        y = y[:, None]

    if len(x) != len(y):
        raise ValueError("Sample counts must match.")

    n = len(x)

    if n == 0:
        raise ValueError("Empty input.")

    x_states = [tuple(row) for row in x]
    y_states = [tuple(row) for row in y]

    count_x = Counter(x_states)
    count_y = Counter(y_states)
    count_xy = Counter(zip(x_states, y_states))

    mi = 0.0

    for (sx, sy), count in count_xy.items():
        pxy = count / n
        px = count_x[sx] / n
        py = count_y[sy] / n

        mi += pxy * np.log2(pxy / (px * py))

    return max(0.0, float(mi))


def _compute_individual_mi(args):
    """
    Worker executed in a separate process.
    """
    i, shm_name, shape, dtype_str = args

    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        data = np.ndarray(
            shape,
            dtype=np.dtype(dtype_str),
            buffer=shm.buf
        )

        x_i = data[:-1, i]
        y = data[1:]

        mi = mutual_information(x_i, y)

        return i, mi

    finally:
        shm.close()


class WMSCalculator:
    def __init__(
        self,
        T: int | None = None,
        eps: float = 1e-12,
        show_progress: bool = True
    ):
        """
        T: Requested number of processes.
           None -> use half of available CPUs.

        eps: Numerical tolerance.
        show_progress: Enable progress bar.
        """

        # Number of logical CPUs available to this process
        if hasattr(os, "process_cpu_count"):
            cpu_count = os.process_cpu_count()
        elif hasattr(os, "sched_getaffinity"):
            cpu_count = len(os.sched_getaffinity(0))
        else:
            cpu_count = os.cpu_count()

        cpu_count = cpu_count or 1

        # Maximum: half of available logical CPUs
        max_processes = max(1, cpu_count // 2)

        if T is not None and T < 1:
            raise ValueError("T must be >= 1.")

        self.T = min(T, max_processes) if T is not None else max_processes

        self.eps = eps
        self.show_progress = show_progress

        print(f"Available logical CPUs: {cpu_count}")
        print(f"Maximum allowed processes: {max_processes}")
        print(f"Selected processes: {self.T}")


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

        data = np.ascontiguousarray(data)

        num_samples, N = data.shape

        # =====================================
        # Step 1: Joint Mutual Information
        # =====================================

        if self.show_progress:
            print("Computing joint mutual information...")

        A = mutual_information(
            data[:-1],
            data[1:]
        )

        # =====================================
        # Step 2: Shared Memory
        # =====================================

        shm = shared_memory.SharedMemory(
            create=True,
            size=data.nbytes
        )

        try:
            shared_data = np.ndarray(
                data.shape,
                dtype=data.dtype,
                buffer=shm.buf
            )

            shared_data[:] = data

            worker_args = [
                (
                    i,
                    shm.name,
                    data.shape,
                    data.dtype.str
                )
                for i in range(N)
            ]

            individual_mis = np.zeros(
                N,
                dtype=np.float64
            )

            # =====================================
            # Step 3: Parallel MI Computation
            # =====================================

            with ProcessPoolExecutor(
                max_workers=min(self.T, N)
            ) as executor:

                futures = [
                    executor.submit(
                        _compute_individual_mi,
                        args
                    )
                    for args in worker_args
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

        finally:
            shm.close()
            shm.unlink()

        # =====================================
        # Step 4: WMS Reward
        # =====================================

        wms = A - B

        reward = (
            A / (A + B)
            if A + B > self.eps
            else 0.5
        )

        return WMSResult(
            reward=float(reward),
            joint_mi=float(A),
            individual_mi_sum=float(B),
            wms=float(wms)
        )


# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    num_samples = 10000
    N = 4096
    T = 100  # Number of parallel processes

    data = rng.integers(
        0, 2,
        size=(num_samples, N),
        dtype=np.uint8
    )

    calculator = WMSCalculator(
        T=None,
        show_progress=True
    )

    start = time.perf_counter()
    result = calculator.calculate(data)
    elapsed = time.perf_counter() - start

    print("\n========== WMS RESULTS ==========")
    print(f"Processes:     {T}")
    print(f"Joint MI:      {result.joint_mi:.6f} bits")
    print(f"Individual MI: {result.individual_mi_sum:.6f} bits")
    print(f"WMS:           {result.wms:.6f} bits")
    print(f"WMS Reward:    {result.reward:.6f}")
    print(f"Elapsed Time:  {elapsed:.4f} seconds")
