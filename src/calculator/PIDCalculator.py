
from dataclasses import dataclass
from collections import Counter
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from dit import Distribution
from dit.pid import PID_WB


BinaryArray: TypeAlias = NDArray[np.integer]
PIDNode: TypeAlias = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PIDResult:
    synergy: float
    redundancy: float
    uniqueness: float
    mixed: float
    total_mi: float


class PIDCalculator:
    """
    Calculate normalized PID values for a binary time series.

    Input:
        data: ndarray of shape (T, N)

    Sources:
        X_t = (X_t^1, ..., X_t^N)

    Target:
        X_(t+1) = (X_(t+1)^1, ..., X_(t+1)^N)

    Mixed atoms are excluded from synergy,
    redundancy, and uniqueness.
    """

    def __init__(self, max_sources: int = 5, tol: float = 1e-9) -> None:
        self.max_sources: int = max_sources
        self.tol: float = tol

    @staticmethod
    def _classify_node(node: PIDNode) -> str:
        groups: list[frozenset[int]] = [
            frozenset(group) for group in node
        ]

        if len(groups) == 1:
            return ("uniqueness" if len(groups[0]) == 1 else "synergy")

        if all(len(group) == 1 for group in groups):
            return "redundancy"

        return "mixed"

    def compute(self, data: BinaryArray) -> PIDResult:
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError("Input data must have shape (T, N).")

        T, N = data.shape

        if T < 2 or N < 2:
            raise ValueError("Require T >= 2 and N >= 2.")

        if N > self.max_sources:
            raise ValueError(f"N={N} exceeds max_sources="f"{self.max_sources}.")

        if not np.all((data == 0) | (data == 1)):
            raise ValueError("Input must contain only binary values.")

        # Build joint distribution:
        # (X_t, X_(t+1))
        samples: list[tuple[int, ...]] = [
            tuple(map(int, np.concatenate((data[t], data[t + 1]))))
            for t in range(T - 1)
        ]

        counts: Counter[tuple[int, ...]] = Counter(samples)
        outcomes: list[tuple[int, ...]] = sorted(counts.keys())

        probs: list[float] = [
            count / (T - 1)
            for count in (counts[outcome] for outcome in outcomes)
        ]

        dist = Distribution(outcomes, probs)

        # Complete Williams-Beer PID
        pid = PID_WB(
            dist,
            sources=[[i] for i in range(N)],
            target=list(range(N, 2 * N))
        )

        totals: dict[str, float] = {
            "synergy": 0.0,
            "redundancy": 0.0,
            "uniqueness": 0.0,
            "mixed": 0.0
        }

        for node in pid._lattice:
            value: float = float(pid.get_pi(node))

            if abs(value) < self.tol:
                value = 0.0

            category: str = self._classify_node(node)
            totals[category] += value

        total_mi: float = sum(totals.values())

        # Ratios are undefined if MI is zero.
        if total_mi <= self.tol:
            return PIDResult(
                synergy=float("nan"),
                redundancy=float("nan"),
                uniqueness=float("nan"),
                mixed=float("nan"),
                total_mi=total_mi
            )

        return PIDResult(
            synergy=totals["synergy"] / total_mi,
            redundancy=totals["redundancy"] / total_mi,
            uniqueness=totals["uniqueness"] / total_mi,
            mixed=totals["mixed"] / total_mi,
            total_mi=total_mi
        )


if __name__ == "__main__":

    rng = np.random.default_rng(42)
    T: int = 10000
    N: int = 3
    data: BinaryArray = rng.integers(low=0, high=2, size=(T, N))

    calculator = PIDCalculator(max_sources=5)
    result: PIDResult = calculator.compute(data)

    print(f"Synergy Ratio:    {result.synergy:.6f}")
    print(f"Redundancy Ratio: {result.redundancy:.6f}")
    print(f"Uniqueness Ratio: {result.uniqueness:.6f}")
    print(f"Mixed Ratio:      {result.mixed:.6f}")
    print(f"Total MI:         {result.total_mi:.6f} bits")

    ratio_sum: float = (
        result.synergy
        + result.redundancy
        + result.uniqueness
        + result.mixed
    )

    print(f"Sum of Ratios:    {ratio_sum:.6f}")
