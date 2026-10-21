
from itertools import combinations
from omegaid.core.decomposition import calc_phiid_multivariate
import numpy as np 
import pandas as pd 

rng = np.random.default_rng(42)
# 5 components, 10000 observations
data = rng.integers(0, 2, size=(5, 10000))

results = []

for i, j in combinations(range(5), 2):
    pair_data = data[[i, j], :]
    
    pair_atoms, _ = calc_phiid_multivariate(
        pair_data,
        pair_data,
        tau=1,
        kind="discrete",
        redundancy="MMI"
    )

    results.append({
        "pair": (i, j),
        "persistent_synergy": np.mean(pair_atoms["sts"]),
        "persistent_redundancy": np.mean(pair_atoms["rtr"]),
        "source_synergy": sum(
            np.mean(v) for k, v in pair_atoms.items()
            if k.startswith("st")
        ),
        "source_redundancy": sum(
            np.mean(v) for k, v in pair_atoms.items()
            if k.startswith("rt")
        )
    })

df = pd.DataFrame(results)
print(df)
