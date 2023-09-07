""" @author: Guillaume Plateau, TAU Systems """

import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from typing import NamedTuple


BinningResults = NamedTuple('BinningResults', avg_x=np.ndarray, avg_y=np.ndarray, std_x=np.ndarray, std_y=np.ndarray,
                            near_ix=np.ndarray, indexes=list, bins=np.ndarray)


def unsupervised_binning(x_data: np.ndarray, y_data: np.ndarray, n_bins_min: int = 3) -> BinningResults:
    # sort by x
    x_permutations = np.argsort(x_data)
    x_data = x_data[x_permutations]
    y_data = y_data[x_permutations]

    # sorted dx, and d(dx)
    d1x: np.ndarray = x_data[1:] - x_data[:-1]
    if all(d1x == 0):
        return BinningResults(avg_x=np.empty((0,)), avg_y=np.empty((0,)), std_x=np.empty((0,)), std_y=np.empty((0,)),
                              near_ix=np.empty((0,)), indexes=[], bins=np.empty((0,)))

    d1x_permutations = np.argsort(d1x)
    d1x = d1x[d1x_permutations]

    d2x: np.ndarray = d1x[1:] - d1x[:-1]
    d2x_permutations = np.argsort(d2x)

    # reset dx
    d1x_inv_permutations = np.argsort(d1x_permutations)
    d1x = d1x[d1x_inv_permutations]

    # identify scan jumps
    best_n_steps = best_pos = 0
    mark_x: np.ndarray = np.array((0,))
    for pos in reversed(d2x_permutations):
        if pos+1 >= d1x_permutations.size:
            continue

        # pos_v1 = d1x_permutations[pos]  # highest value among small spaces
        pos_v2 = d1x_permutations[pos+1]  # smallest jump
        mark_x, = np.nonzero(d1x >= d1x[pos_v2])  # all jump positions

        if mark_x.size + 1 >= n_bins_min:
            break
        elif mark_x.size + 1 > best_n_steps:
            best_n_steps = mark_x.size + 1
            best_pos = pos

        if pos == d2x_permutations[0]:
            # pos_v1 = d1x_permutations[best_pos]
            pos_v2 = d1x_permutations[best_pos + 1]
            mark_x, = np.nonzero(d1x >= d1x[pos_v2])

    # bins limits
    bins = np.array([(x_data[m] + x_data[m+1]) / 2 for m in mark_x])
    bins = np.insert(bins, 0, x_data[0] - 1)
    bins = np.append(bins, x_data[-1] + 1)

    # bin the dataset
    avg_x = np.zeros((bins.size-1,))
    avg_y = np.zeros((bins.size-1,))
    std_x = np.zeros((bins.size-1,))
    std_y = np.zeros((bins.size-1,))
    near_ix = np.zeros((bins.size-1,))
    indexes: list[np.ndarray] = []

    for it, (lim_low, lim_high) in enumerate(zip(bins[:-1], bins[1:])):
        pts_to_bin, = np.nonzero((lim_low <= x_data) & (x_data < lim_high))

        indexes.append(x_permutations[pts_to_bin])
        avg_x[it] = np.mean(x_data[pts_to_bin])
        avg_y[it] = np.mean(y_data[pts_to_bin])
        std_x[it] = np.std(x_data[pts_to_bin])
        std_y[it] = np.std(y_data[pts_to_bin])

        # find best representative
        d_rep = np.abs(y_data[pts_to_bin] - avg_y[it])
        pos_rep, = np.nonzero(d_rep == np.min(d_rep))
        near_ix[it] = x_permutations[pts_to_bin[pos_rep[0]]]

    avg_bin = np.mean((bins[1:] - bins[:-1])[1:-1])
    bins[0] = min(bins[0],  bins[1] - avg_bin)
    bins[-1] = max(bins[-1], bins[-2] + avg_bin)

    return BinningResults(avg_x, avg_y, std_x, std_y, near_ix, indexes, bins)


if __name__ == '__main__':
    n_steps = 13
    n_pts = 40
    err_x = 16.
    n_min = n_steps

    base_x = np.linspace(-450., 30., n_steps)
    x: np.ndarray
    x, _ = np.meshgrid(base_x, np.arange(n_pts))
    x += err_x * (2 * np.random.random((n_pts, n_steps)) - 1)
    x = x.transpose().reshape((x.size,))

    c_y = [12.5905, 0.048984, 0.00072434, 8.6414e-06, 3.9397e-08, 5.1607e-11]
    y = polyval(x, c_y)

    _bins = unsupervised_binning(x, y, n_min)

    plt.figure(figsize=(3.2, 2.4))
    plt.plot(x, y, '.c')
    plt.errorbar(_bins[0], _bins[1], yerr=_bins[3], xerr=_bins[2], c='k', alpha=0.66, linestyle='None')
    for lim in _bins[6]:
        plt.axvline(lim, color='k', linestyle='--', linewidth=0.5)
    plt.show(block=True)

    if _bins[0].size == base_x.size:
        print(f'dx:\n\t{_bins[0] - base_x}')
        print(f'dy:\n\t{_bins[1] - polyval(base_x, c_y)}')
        print(f'bins:\n\t{_bins[6]}')

    print('done')
