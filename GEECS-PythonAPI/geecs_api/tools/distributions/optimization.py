""" @author: Guillaume Plateau, TAU Systems """

import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from typing import Optional


def gradient_descent(eval_function, max_iterations: int = 10, threshold: float = 0.5,
                     guess: Optional[np.ndarray] = None, spans: Optional[np.ndarray] = None,
                     learning_rate: float = 0.05, momentum: float = 0.8):
    setpoints: np.ndarray

    if spans is not None:
        if guess is None:
            setpoints = np.mean(spans, axis=1)  # centers of the ranges
        else:
            np.clip(p, t[:, 0], t[:, 1])

    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10

    while i < max_iterations and diff > threshold:
        delta_w = -learning_rate * grad_func(w, extra_param) + momentum * delta_w
        w = w + delta_w

        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, extra_param)))

        # update iteration number and diff between successive values
        # of objective function
        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history



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
