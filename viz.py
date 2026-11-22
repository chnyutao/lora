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
    'cola': ((0, (2, 1)), 'o'),
    'qnli': ((0, (4, 1)), 's'),
    'sst2': (':', '^'),
}

# initialize figure
ax = plt.figure(figsize=(4, 3)).gca()
xticks = np.array([0, 2, 100, 200, 400, 800])
ax.set_xlim(0**0.5, 800**0.5)
ax.set_xticks(xticks**0.5, labels=xticks)
ax.set_ylim(0.98 - 1e-3, 1.00 + 1e-3)
ax.set_yticks([0.98, 0.985, 0.99, 0.995, 1])
ax.set_yticklabels(['98', '', '99', '', '100'])
for threshold, n in zip((0.99, 0.999), (2, 100)):
    kwds = {'linestyle': ':', 'color': 'tab:gray'}
    ax.axhline(threshold, c='gray', ls=':')
    ax.vlines(n**0.5, ymin=0, ymax=threshold, color='gray', ls=':')
ax.set_xlabel('# Leading Singular Values')
ax.set_ylabel('Cumulative Energy Ratio (%)')

# visualize energy distribution of dataset
for dataset, (linestyle, marker) in styles.items():
    # read singular values
    config.dataset.name = dataset
    paths = Assets.sv(config).glob('*.pt')
    arrays = torch.stack([torch.load(path) for path in paths])
    assert torch.isfinite(arrays).all() and (arrays > 0).all()
    # compute spectral energy
    energy = arrays**2
    energy = energy.cumsum(dim=1) / energy.sum(dim=1, keepdim=True)
    energy = energy.mean(dim=0)
    # plot cumulative energy ratio
    x = np.arange(len(energy)) + 1
    ax.plot(
        *(x**0.5, energy),
        label=dataset,
        linestyle=linestyle,
        linewidth=1.5,
        marker=marker,
        markerfacecolor='white',
        markersize=4,
        markevery=np.concat([np.arange(2), np.arange(20, len(x), 50)]),
    )
ax.legend(loc='lower right')
plt.gcf().tight_layout()
plt.savefig('asset/out.pdf', bbox_inches='tight')
