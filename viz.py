from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from config import Config
from utils import Assets

# configuration
config = tyro.cli(Config)
styles = {
    'cola': {'ls': (0, (4, 1)), 'marker': 'o', 'ms': 3},
    'qnli': {'ls': (0, (2, 1)), 'marker': 's', 'ms': 2.5},
    'sst2': {'ls': (0, (1, 1)), 'marker': '^', 'ms': 3},
}

# initialize figure
ax = plt.figure(figsize=(4, 3)).gca()
model = config.model.name.lower()
if model.startswith('google-t5/'):
    xticks = np.array([0, 2, 100, 200, 400, 800])
    yticks = np.array([98, 98.5, 99, 99.5, 100])
    thresholds = [(2, 100), (99, 99.9)]
elif model.startswith('qwen/'):
    xticks = np.array([0, 5, 200, 400, 800, 1600])
    yticks = np.array([80, 85, 90, 95, 100])
    thresholds = [(5, 200), (90, 97.5)]
else:
    raise NotImplementedError
ax.set_xlim(xticks[0] ** 0.5, xticks[-1] ** 0.5)
ax.set_xticks(xticks**0.5, labels=xticks)
pad = (yticks[-1] - yticks[0]) / 20
ax.set_ylim(yticks[0] - pad, yticks[-1] + pad)
ax.set_yticks(yticks)
ax.set_yticklabels(['' if i % 2 else f'{y:.0f}' for i, y in enumerate(yticks)])
for x, y in zip(*thresholds):
    kwds = {'alpha': 0.5, 'color': 'gray', 'ls': ':', 'lw': 1}
    ax.vlines(x**0.5, ymin=0, ymax=y, **kwds)
    ax.axhline(y, **kwds)
ax.set_xlabel('# Leading Singular Values')
ax.set_ylabel('Cumulative Energy Ratio (%)')

# visualize dataset spetral energy
for dataset, kwds in styles.items():
    # read singular values
    config.dataset.name = dataset
    paths = Assets.sv(config).glob('*.pt')
    arrays = torch.stack([torch.load(path) for path in paths])
    assert torch.isfinite(arrays).all() and (arrays > 0).all()
    # compute spectral energy
    energy = arrays**2
    energy = energy.cumsum(dim=1) / energy.sum(dim=1, keepdim=True)
    energy = energy.mean(dim=0) * 100
    # plot cumulative energy ratio
    x = np.arange(len(energy)) + 1
    indices = np.concat([np.arange(2), np.arange(20, len(x), 50)])
    kwds.update({'markevery': indices, 'mfc': 'w', 'lw': 1.5})
    ax.plot(x**0.5, energy, label=dataset, **kwds)
ax.legend(loc='lower right')
plt.gcf().tight_layout()
plt.savefig('asset/out.pdf', bbox_inches='tight')
