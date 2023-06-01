""" @author: Guillaume Plateau, TAU Systems """

import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt


def find(condition: np.ndarray):
    """
    vector = [1, 2, 3, 1, 2, 3]
    [2, 5] = find(vector > 2)
    """
    ret = np.array(range(len(condition)))

    return ret[condition]


def bin_scan(x_data: np.ndarray, y_data: np.ndarray, n_bins_min: int = 3):
    # sort by x
    x_permutations = np.argsort(x_data)
    x_data = x_data[x_permutations]
    y_data = y_data[x_permutations]

    # sorted dx, and d(dx)
    d1x: np.ndarray = x_data[1:] - x_data[:-1]
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
        if pos+1 >= len(d1x_permutations):
            continue

        # pos_v1 = d1x_permutations[pos]  # highest value among small spaces
        pos_v2 = d1x_permutations[pos+1]  # smallest jump
        mark_x = find(d1x >= d1x[pos_v2])  # all jump positions

        if mark_x.size + 1 >= n_bins_min:
            break
        elif mark_x.size + 1 > best_n_steps:
            best_n_steps = mark_x.size + 1
            best_pos = pos

        if pos == d2x_permutations[0]:
            # pos_v1 = d1x_permutations[best_pos]
            pos_v2 = d1x_permutations[best_pos + 1]
            mark_x = find(d1x >= d1x[pos_v2])

    # bins limits
    bin_lim_x = [(x_data[m] + x_data[m+1]) / 2 for m in mark_x]
    bin_lim_x.insert(0, x_data[0] - 1)
    bin_lim_x.append(x_data[-1] + 1)

    # bin the dataset
    bin_x = np.zeros((len(bin_lim_x)-1,))
    avg_y = np.zeros((len(bin_lim_x)-1,))
    std_x = np.zeros((len(bin_lim_x)-1,))
    std_y = np.zeros((len(bin_lim_x)-1,))
    near_ix = np.zeros((len(bin_lim_x)-1,))
    indexes = []

    for it, (lim_low, lim_high) in enumerate(zip(bin_lim_x[:-1], bin_lim_x[1:])):
        t_low = lim_low <= x_data
        t_high = x_data < lim_high
        pts_to_bin = find(np.array([lc & hc for lc, hc in zip(t_low, t_high)]))

        indexes.append(x_permutations[pts_to_bin])
        bin_x[it] = np.mean(x_data[pts_to_bin])
        avg_y[it] = np.mean(y_data[pts_to_bin])
        std_x[it] = np.std(x_data[pts_to_bin])
        std_y[it] = np.std(y_data[pts_to_bin])

        # find best representative
        d_rep = np.abs(y_data[pts_to_bin] - avg_y[it])
        pos_rep = find(d_rep == np.min(d_rep))
        near_ix[it] = x_permutations[pts_to_bin[pos_rep[0]]]

    return bin_x, avg_y, std_x, std_y, near_ix, indexes


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

    # c_y = [5.1607e-11, 3.9397e-08, 8.6414e-06, 0.00072434, 0.048984, 12.5905]
    c_y = [12.5905, 0.048984, 0.00072434, 8.6414e-06, 3.9397e-08, 5.1607e-11]
    y = polyval(x, c_y)

    # plt.figure(figsize=(3.2, 2.4))
    # plt.plot(x, y, '.')
    # plt.show(block=True)

    bins = bin_scan(x, y, n_min)

    plt.figure(figsize=(3.2, 2.4))
    plt.plot(x, y, '.c')
    plt.errorbar(bins[0], bins[1], yerr=bins[3], xerr=bins[2], c='k', alpha=0.66, linestyle='None')
    plt.show(block=True)

    if bins[0].size == base_x.size:
        print(f'dx:\n\t{bins[0] - base_x}')
        print(f'dy:\n\t{bins[1] - polyval(base_x, c_y)}')

    print('done')
