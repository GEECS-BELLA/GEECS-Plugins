# Pythonic module for kHzwave project

[Project Home](https://github.com/radiasoft/khzwave_python/)

## WFS20 driver wrapper

This module consists of three major components:

* Network server/client to transport image data across a UDP socket connection
* `ctypes` driver wrapper for `WFS.h` and `visatype.h` that automatically populates correct arguments and results types to interface with higher-level Python code
* WFS20 camera control class that can interact with the driver with reasonable name in a Pythonic manner

## Installation

For ease of management, I've been using the miniforge3 Anaconda Python distribution. This
module should work with any Python distribution with access to numpy and pyzmq.

Like most Python modules, you can use:

`pip install .`

or to reinstall over the old version:

`pip install --force .`

## Performance

A simple performance gain should be possible by `cythonizing` the wfs.py module using
cython. This is not currently a concern
