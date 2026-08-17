"""Parallel, filesystem-efficient equivalent of Causal-Adapter/pendulum.py."""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def projection(phi: float, x: float, y: float, base: float = -0.5) -> float:
    b = y - x * math.tan(phi)
    return (base - b) / math.tan(phi)


def render(task):
    index, i, j, root = task
    theta = i * math.pi / 200.0
    phi = j * math.pi / 200.0
    x = 10 + 8 * math.sin(theta)
    y = 10.5 - 8 * math.cos(theta)
    light = projection(phi, 10, 10.5, 20.5)
    ball_x = 10 + 9.5 * math.sin(theta)
    ball_y = 10.5 - 9.5 * math.cos(theta)
    p0 = projection(phi, 10.0, 10.5)
    p1 = projection(phi, ball_x, ball_y)
    mid = (p0 + p1) / 2
    shade = max(3, abs(p0 - p1))

    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.add_artist(plt.Polygon(([10, 10.5], [x, y]), color='black', linewidth=3))
    ax.add_artist(plt.Circle((x, y), 1.5, color='firebrick'))
    ax.add_artist(plt.Circle((light, 20.5), 3, color='orange'))
    ax.add_artist(plt.Polygon(([mid - shade / 2, -0.5], [mid + shade / 2, -0.5]), color='black', linewidth=3))
    ax.set_xlim((0, 20)); ax.set_ylim((-1, 21)); ax.axis('off')
    # Matches the original script: first test sample at index 4, then every 4th.
    split = 'test' if index >= 4 and index % 4 == 0 else 'train'
    name = f'a_{i}_{j}_{shade}_{mid}.png'
    fig.savefig(Path(root) / split / name, dpi=96)
    plt.close(fig)
    return split


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    root = Path(args.output)
    (root / 'train').mkdir(parents=True, exist_ok=True)
    (root / 'test').mkdir(parents=True, exist_ok=True)
    pairs = [(i, j) for i in range(-40, 44) for j in range(60, 148) if j != 100]
    tasks = [(k, i, j, str(root)) for k, (i, j) in enumerate(pairs)]
    with mp.get_context('spawn').Pool(args.workers) as pool:
        counts = {'train': 0, 'test': 0}
        for n, split in enumerate(pool.imap_unordered(render, tasks, chunksize=4), 1):
            counts[split] += 1
            if n % 250 == 0:
                print(n, counts, flush=True)
    print('complete', counts, flush=True)


if __name__ == '__main__':
    main()
