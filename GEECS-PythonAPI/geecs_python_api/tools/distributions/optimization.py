""" @author: Guillaume Plateau, TAU Systems """

import numpy as np
import pandas as pd
from xopt import Xopt
from xopt.evaluator import Evaluator
from xopt.vocs import VOCS
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from typing import Optional, Union
import matplotlib.pyplot as plt
from scipy.optimize import minimize


if __name__ == '__main__':
    history = []
    costs = []

    # def curve(v):
    #     value = np.sqrt((v[0] - 2.) ** 2 + (v[1] + 1.) ** 2)
    #     if not isinstance(history, np.ndarray):
    #         history.append(v)
    #         costs.append(value)
    #     return value

    def curve(input_dict):
        v = input_dict["x"]
        value = np.sqrt((v[0] - 2.) ** 2 + (v[1] + 1.) ** 2)
        return {"f": value}

    v0 = np.array([0., 0.])

    YAML = """
    xopt: {}
    generator:
      name: neldermead
      initial_point: {x0: 0., x1: 0.}
      adaptive: true
      xatol: 0.0001
      fatol: 0.0001  
    evaluator:
      function: curve
    vocs:
      variables:
        x0: [-5, 5]
        x1: [-5, 5]
      objectives: {y: MINIMIZE}
    """

    # define variables and function objectives
    vocs = VOCS.from_yaml(YAML)
    print(vocs.dict())

    evaluator = Evaluator(function=curve)
    generator = UpperConfidenceBoundGenerator(vocs)

    X_yaml = Xopt(evaluator=evaluator)
    X_min = [2, -1]
    X_yaml.evaluate({"x0": X_min[0], "x1": X_min[1]})

    # res = minimize(curve, v0, method='Nelder-Mead', tol=1e-6)

    history = np.array(history)
    costs = np.array(costs)

    plt.figure(figsize=(3.2, 2.4))
    xv = np.linspace(-5., 5., 101)
    yv = np.linspace(-5., 5., 101)
    X, Y = np.meshgrid(xv, yv)
    Z = curve(np.stack([X, Y]))
    plt.imshow(Z, aspect='equal', extent=[-5, 5, 5, -5], origin='upper')
    plt.plot(history[:, 0], history[:, 1], marker='.', color='black')
    plt.plot(history[-1, 0], history[-1, 1], marker='o', color='red')
    plt.show(block=True)

    print('done')
