""" @author: Guillaume Plateau, TAU Systems """

import os
import re
import numpy as np
import shelve
from scipy.io import savemat, loadmat
from pathlib import Path
from typing import Any, Optional
import tkinter as tk
from tkinter import filedialog
# import htu_scripts.analysis.undulator_no_scan as uns


# Pop-ups initialization
tk_root = tk.Tk()
tk_root.withdraw()


def save_py(file_path: Optional[Path] = None, data: Optional[dict[str, Any]] = None):
    if not file_path and data:
        file_path = filedialog.asksaveasfilename(defaultextension='',
                                                 filetypes=[('All Files', '*.*'), ('Shelve Files', '*.dat')],
                                                 initialdir=Path.home(),
                                                 title='Export as Python Shelf:')

    if (not file_path) or (not data):
        return
    else:
        file_path = Path(file_path)

    with shelve.open(str(file_path), 'c') as shelve_file:
        for key, value in data.items():
            try:
                shelve_file[key] = value
            except Exception as ex:
                print(f'Failed to write "{key}" data to shelve file')
                print(ex)
                continue


def load_py(file_path: Optional[Path] = None, variables: Optional[list[str]] = None):
    if file_path and not re.search(r'\.[^\.]+$', str(file_path)):
        file_path = Path(f'{file_path}.dat')

    if (not file_path) or (not file_path.is_file()):
        file_path = filedialog.askopenfilename(defaultextension='.dat',
                                               filetypes=[('All Files', '*.*'), ('Shelve Files', '*.dat')],
                                               initialdir=Path.home(),
                                               title='Open a Python shelf:')

    if not file_path:
        return
    else:
        file_path = os.path.normpath(file_path)
        file_path = re.split(r'\.[^\.]+$', str(file_path))[0]

    with shelve.open(file_path, 'r') as shelve_file:
        for key, value in shelve_file.items():
            if variables:
                if key in variables:
                    globals()[key] = value
            else:
                globals()[key] = value


def save_mat(file_path: Optional[Path] = None, data: Optional[dict[str, Any]] = None):
    if not file_path and data:
        file_path = filedialog.asksaveasfilename(defaultextension='.mat',
                                                 filetypes=[('MATLAB Files', '*.mat'), ('All Files', '*.*')],
                                                 initialdir=Path.home(),
                                                 title='Export as Matlab file:')

    if (not file_path) or (not data):
        return
    else:
        file_path = Path(file_path)

    savemat(file_path, data)


def load_mat(file_path: Optional[Path] = None, variables: Optional[list[str]] = None):
    if (not file_path) or (not file_path.is_file()):
        file_path = filedialog.askopenfilename(defaultextension='.mat',
                                               filetypes=[('MATLAB Files', '*.mat'), ('All Files', '*.*')],
                                               initialdir=Path.home(),
                                               title='Select a Matlab file:')
    if not file_path:
        return None, None
    else:
        file_path = Path(file_path)

    if not variables:
        data = loadmat(str(file_path))
    else:
        data = loadmat(str(file_path), variable_names=variables)

    for key, value in data.items():
        if not key.startswith('__'):
            globals()[key] = value


def save_csv(file_name=None, file_dir=None, data=None, header='', data_format='%.3f', delimiter='\t'):
    if not file_name:
        file_name = 'py3_data.csv'

    if not file_dir:
        file_dir = filedialog.askdirectory(title='Target folder for .csv file:')

    if file_dir:
        file_path = os.path.normpath(os.path.join(file_dir, file_name))
        np.savetxt(file_path, data, delimiter=delimiter, fmt=data_format, header=header, comments="")


if __name__ == '__main__':
    a_string = 'hello!'
    a_list = [a_string, 'people']
    a_numeric = 1.23456789
    an_array = np.random.random((3, 4))
    a_dict = {'str': a_string, 'lst': a_list, 'num': a_numeric, 'arr': an_array}

    _data = {'str': a_string, 'lst': a_list, 'num': a_numeric, 'arr': an_array, 'dct': a_dict}

    # test_file = Path(r'C:\Users\GuillaumePlateau\Desktop\tests\test.mat')
    test_file = None

    # save_mat(test_file, data=_data)
    # load_mat(test_file, variables=['str', 'lst', 'num', 'arr', 'dct'])

    # test_file = Path(test_file.__str__()[:-4])
    save_py(test_file, data=_data)
    load_py(test_file, variables=['str', 'lst', 'num', 'arr', 'dct'])

    # load_py(Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\05-May\23_0509\analysis\
    # Scan028\UC_VisaEBeam7\profiles_analysis.dat'))
    # noinspection PyUnboundLocalVariable
    # average_analysis = uns.UndulatorNoScan.is_image_valid(average_analysis, 2.)

    print('done')
