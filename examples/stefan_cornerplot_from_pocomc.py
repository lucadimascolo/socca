import numpy as np

import pickle
with open("run_pocomc_dump/pmc_final.state", "rb") as f:
    sampler = pickle.load(f)

results = sampler['particles'].compute_results()
samples = results['x'][-1]
logweights = results['logw'][-200:]
weights = np.exp(logweights - logweights.max())
weights /= weights.sum()

import corner
import matplotlib.pyplot as plt

labels = ['xc', 'yc', 'theta', 're_disk', 'Ie_disk', 'inc', 'zs', 
          're_bar', 'Ie_bar', 'ns_bar', 'rot_bar', 'e_bar']

corner.corner(samples, weights=weights, labels=labels)
plt.savefig("test_corner.pdf", dpi=300)
plt.close()