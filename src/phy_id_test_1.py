
import numpy as np
from omegaid.core.decomposition import calc_phiid_multivariate

rng = np.random.default_rng(42)

# 5 components, 10000 observations
data = rng.standard_normal((5, 10000))

atoms, details = calc_phiid_multivariate(
    data,
    data,
    tau=1
)

print(atoms)
