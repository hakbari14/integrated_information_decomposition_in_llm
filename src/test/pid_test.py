
import numpy as np
import pandas as pd
import dit
from collections import Counter
from dit.pid import PID_WB


def calculate_pid(data, tau=1):
    """
    Calculate PID for a binary multivariate time series.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, N), where T is the number of time steps
        and N is the number of binary components.

    tau : int
        Time lag between source and target.

    Returns
    -------
    pid : dit PID object
    atoms_df : pd.DataFrame
    """

    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("data must have shape (T, N)")

    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("Data must contain only 0 and 1")

    T, N = data.shape

    if tau < 1 or tau >= T:
        raise ValueError("tau must be between 1 and T-1")

    # Sources at time t
    X_t = data[:-tau]

    # Target: full system at time t+tau
    X_future = data[tau:]

    # Convert future binary vector to a categorical state
    weights = 2 ** np.arange(N - 1, -1, -1)
    Y = X_future @ weights

    # Build empirical joint distribution
    samples = [
        tuple(map(int, list(x) + [y]))
        for x, y in zip(X_t, Y)
    ]

    counts = Counter(samples)
    total = len(samples)

    outcomes = list(counts.keys())
    probabilities = [
        counts[o] / total for o in outcomes
    ]

    d = dit.Distribution(outcomes, probabilities)

    # N source variables and one target
    inputs = [(i,) for i in range(N)]
    output = (N,)

    # Williams-Beer PID
    pid = PID_WB(
        d,
        sources=inputs,
        target=output
    )

    # Extract all PID atoms
    results = []

    for node in pid._lattice:
        value = float(pid.get_pi(node))

        results.append({
            "atom": str(node),
            "value": value
        })

    atoms_df = pd.DataFrame(results)

    return pid, atoms_df




def extract_pid_components(pid):
    # Common redundancy among all 3 sources
    redundancy = pid.get_pi(
        ((0,), (1,), (2,))
    )

    # Unique information of each source
    unique = {
        f"U{i+1}": pid.get_pi(((i,),))
        for i in range(3)
    }

    # Pure three-way synergy
    synergy = pid.get_pi(
        ((0, 1, 2),)
    )

    return {
        "redundancy_3way": float(redundancy),
        **{k: float(v) for k, v in unique.items()},
        "synergy_3way": float(synergy)
    }


# ----------------------------------
# Example: 3 binary components
# ----------------------------------

rng = np.random.default_rng(42)

T = 10000
N = 3

data = rng.integers(
    low=0,
    high=2,
    size=(T, N)
)

pid, atoms_df = calculate_pid(data, tau=1)

print("PID atoms:")
print(atoms_df.to_string(index=False))

print("\nTotal PID information:")
print(atoms_df["value"].sum())

components = extract_pid_components(pid)

for name, value in components.items():
    print(f"{name}: {value:.6f}")

