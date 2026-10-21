import numpy as np
from omegaid.core.decomposition import calc_phiid_multivariate
from omegaid.utils.backend import set_backend

# For programmatic control, you can also use set_backend
# set_backend('cupy')

# Generate some random time series data (4 sources, 2 targets)
n_sources = 4
n_targets = 2
n_samples = 10000
tau = 1
sources = np.random.randn(n_sources, n_samples)
targets = np.random.randn(n_targets, n_samples)

# Calculate Phi-ID using the Doublet Lattice approximation
atoms_res, _ = calc_phiid_multivariate(sources, targets, tau)

# Print a synergistic atom, e.g., between source 0 and target 1
print("Synergy (s0, t1):", atoms_res.get((0, 1), "N/A"))