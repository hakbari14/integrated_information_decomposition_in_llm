
import numpy as np
import pandas as pd

from omegaid.core.decomposition import calc_phiid_multivariate


def calculate_phiid(data, tau=1):
    """
    PhiID for a multivariate binary time series.

    Parameters
    ----------
    data : ndarray, shape (T, N)
        Binary time series (0/1).

    tau : int
        Time lag.

    Returns
    -------
    atoms : dict
        PhiID decomposition results.
    details : dict
        Intermediate calculations.
    """

    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("Expected data shape (T, N)")

    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("Data must be binary")

    T, N = data.shape

    if T <= tau or tau < 1:
        raise ValueError("Invalid tau")

    # omegaid expects (N, T)
    signals = data.T.astype(float)

    atoms, details = calc_phiid_multivariate(
        signals,
        signals,
        tau=tau
    )

    return atoms, details


def summarize_phiid(atoms):
    """
    Summarize returned atoms without assuming
    a particular key format.
    """

    rows = []

    for key, values in atoms.items():
        arr = np.asarray(values, dtype=float)

        rows.append({
            "atom": str(key),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr))
        })

    return pd.DataFrame(rows)


# -----------------------------------
# Example: 3-component binary series
# -----------------------------------

rng = np.random.default_rng(42)

T = 10000
N = 3

# Example with temporal dependencies
data = np.zeros((T, N), dtype=int)
data[0] = rng.integers(0, 2, size=N)

for t in range(1, T):
    data[t, 0] = data[t-1, 0]
    data[t, 1] = data[t-1, 0] ^ data[t-1, 2]
    data[t, 2] = data[t-1, 1]

atoms, details = calculate_phiid(data, tau=1)

summary = summarize_phiid(atoms)

print("PhiID atoms:")
print(summary.to_string(index=False))

print("\nRaw atom keys:")
print(list(atoms.keys()))
