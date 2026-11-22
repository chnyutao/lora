"""Plot singular values and cumulative spectral energy from per-seed .pt files.

Each run of inference.py writes asset/sv/<model>-<dataset>/<seed>.pt holding
the descending singular values of H. This script globs those files via
utils.get_sv_path, aggregates the seeds, and plots the mean spectrum with
error bars plus the cumulative spectral energy (y-axis zoomed to 90%-100%).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from config import Config
from utils import Assets


def read_values(sv_dir: Path) -> torch.Tensor:
    """Stack per-seed singular value arrays into a seeds x values tensor."""
    try:
        paths = sorted(sv_dir.glob('*.pt'), key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f'expected numeric per-seed filenames in {sv_dir}') from error
    if not paths:
        raise ValueError(f'no singular value files found in {sv_dir}')
    arrays = [torch.load(path, weights_only=True) for path in paths]
    length = arrays[0].shape[0]
    if any(array.ndim != 1 or array.shape[0] != length for array in arrays):
        raise ValueError('all singular value files must hold equal-length vectors')
    matrix = torch.stack(arrays)
    if not torch.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('singular values must be finite and nonnegative')
    return matrix


def plot_values(values: torch.Tensor, output: Path) -> None:
    """Plot the mean spectrum (error bars) and cumulative energy across seeds."""
    if values.ndim != 2:
        raise ValueError('expected a seeds x values matrix')
    means = values.mean(dim=0).numpy()
    stds = values.std(dim=0).numpy()
    indices = np.arange(1, len(means) + 1)

    # per-seed cumulative spectral energy, averaged across seeds
    energy = values**2
    cum_energy = (
        (energy.cumsum(dim=1) / energy.sum(dim=1, keepdim=True)).mean(dim=0).numpy()
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].errorbar(
        indices,
        means,
        yerr=stds,
        marker='o',
        markersize=3,
        linewidth=1,
        capsize=2,
        label='mean singular value',
    )
    axes[0].set_yscale('log')
    axes[0].set_title('Singular values of H across seeds (descending)')
    axes[0].set_xlabel('singular value index')
    axes[0].set_ylabel('mean singular value (log scale)')
    axes[0].grid(alpha=0.3, which='both')
    axes[0].legend()

    axes[1].plot(
        indices,
        cum_energy,
        marker='o',
        markersize=3,
        linewidth=1,
        label='cumulative energy',
    )
    for threshold in (0.95, 0.99):
        k = np.searchsorted(cum_energy, threshold) + 1
        axes[1].axhline(
            threshold,
            color='orange' if threshold == 0.99 else 'green',
            linestyle='--',
            alpha=0.7,
            label=f'{int(threshold * 100)}% energy',
        )
        axes[1].axvline(
            k,
            color='red',
            linestyle=':',
            alpha=0.7,
            label=f'{int(threshold * 100)}% at k={k}',
        )
    axes[1].set_title(
        'Cumulative spectral energy fraction vs number of singular values'
    )
    axes[1].set_xlabel('number of top singular values kept')
    axes[1].set_ylabel('cumulative energy fraction')
    axes[1].set_ylim(0.9, 1.005)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'Saved SVD visualization to {output}')


def main() -> None:
    config = tyro.cli(Config)
    sv_dir = Assets.sv(config)
    values = read_values(sv_dir)
    print(f'Loaded {values.shape[0]} seed(s) with {values.shape[1]} singular values')
    plot_values(values, sv_dir.with_suffix('.png'))


if __name__ == '__main__':
    main()
