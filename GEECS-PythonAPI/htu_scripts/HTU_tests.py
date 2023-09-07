import os
import time
import numpy as np
import pandas as pd
import shelve
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.diagnostics.ebeam_diagnostics import EBeamDiagnostics
from geecs_python_api.controls.devices.HTU.diagnostics import UndulatorStage
from scipy.io import savemat, loadmat
import tkinter as tk
from tkinter import filedialog


# Pop-ups initialization
tk_root = tk.Tk()
tk_root.withdraw()


def save_py(file_path='', data=None):
    if (not file_path) or (not os.path.exists(file_path)):
        file_path = filedialog.asksaveasfilename(defaultextension='.out',
                                                 filetypes=[('All Files', '*.*'), ('Shelve Files', '*.out')],
                                                 initialdir=r'C:\Users\gplateau\Documents\Data\Tmp',
                                                 title='Export as Python Shelf:')

    if (not file_path) or (not data):
        return False
    else:
        file_path = os.path.normpath(file_path)

    # my_shelf = shelve.open(os.path.join(file_path, file_name), 'n')  # 'n' for new
    my_shelf = shelve.open(file_path, 'n')  # 'n' for new

    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))

    my_shelf.close()
    return True


def load_py(file_path=''):
    if (not file_path) or (not os.path.exists(file_path)):
        file_path = filedialog.askopenfilename(defaultextension='.out',
                                               filetypes=[('All Files', '*.*'), ('Shelve Files', '*.out')],
                                               initialdir=r'C:\Users\gplateau\Box\Documents\Projects\Sensus\Data',
                                               title='Open a Python shelf:')

    if not file_path:
        return False
    else:
        file_path = os.path.normpath(file_path)

    shelf = shelve.open(file_path)

    for key in shelf:
        globals()[key] = shelf[key]

    shelf.close()
    return True


def save_csv(file_name=None, file_dir=None, data=None, header='', data_format='%.3f', delimiter='\t'):
    if not file_name:
        file_name = 'py3_data.csv'

    if not file_dir:
        file_dir = filedialog.askdirectory(title='Target folder for .csv file:')

    if file_dir:
        file_path = os.path.normpath(os.path.join(file_dir, file_name))
        np.savetxt(file_path, data, delimiter=delimiter, fmt=data_format, header=header, comments="")


def save_mat(file_name='py3_mat.mat', file_path='', data=None, initial_dir=r'C:\Users\gplateau\Documents\Data\Tmp',
             title='Export as Matlab file:'):
    if (not file_path) or (not os.path.exists(file_path)):
        file_path = filedialog.asksaveasfilename(filetypes=[('MATLAB Files', '*.mat'), ('All Files', '*.*')],
                                                 initialdir=initial_dir,
                                                 title=title)
        if (not file_path) or (not file_name) or (not data):
            return False
    else:
        file_path = os.path.join(file_path, file_name)

    file_path = os.path.normpath(file_path)

    savemat(file_path, data)
    return True


def load_mat(file_path='', variables=None, initial_dir=r'C:\Users\gplateau\Documents\Data\Tmp',
             title='Select a Matlab file:'):
    if (not file_path) or (not os.path.exists(file_path)):
        file_path = filedialog.askopenfilename(filetypes=[('MATLAB Files', '*.mat'), ('All Files', '*.*')],
                                               initialdir=initial_dir,
                                               title=title)
    if not file_path:
        return None, None
    else:
        file_path = os.path.normpath(file_path)

    if not variables:
        data = loadmat(file_path)
    else:
        data = loadmat(file_path, variable_names=variables)

    return data, file_path


# create experiment object
# htu = HtuExp()
# GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
data = GeecsDatabase.collect_exp_info('Undulator')

save_mat(data=GeecsDevice.exp_info)
# e_beam = EBeamDiagnostics()
# velmex = UndulatorStage()
# time.sleep(.1)

# do something
# print(f'Velmex position: {velmex.get_position()}')
# velmex.set_position(station=3, diagnostic='energy')

# close connections
# e_beam.close()
# for controller in e_beam.controllers:
#     controller.close()
# velmex.close()
