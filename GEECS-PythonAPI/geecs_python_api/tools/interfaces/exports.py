""" @author: Guillaume Plateau, TAU Systems """

import os
import re
import numpy as np
import shelve
from scipy.io import savemat, loadmat
from pathlib import Path
from typing import Any, Optional, Union
import tkinter as tk
from tkinter import filedialog


# Pop-ups initialization
tk_root = tk.Tk()
tk_root.withdraw()


def save_py(file_path: Optional[Path] = None, data: Optional[dict[str, Any]] = None, as_bulk: bool = False):
    if not file_path and data:
        file_path = filedialog.asksaveasfilename(defaultextension='',
                                                 filetypes=[('All Files', '*.*'), ('Shelve Files', '*.dat')],
                                                 initialdir=Path.home(),
                                                 title='Export as Python Shelf:')

    if (not file_path) or (not data):
        return
    else:
        file_path = Path(file_path)

    with shelve.open(str(file_path), 'n') as shelve_file:
        if as_bulk:
            try:
                shelve_file['data'] = data
            except Exception as ex:
                print('Failed to write data as bulk to shelve file')
                print(ex)
        else:
            for key, value in data.items():
                try:
                    shelve_file[key] = value
                except Exception as ex:
                    print(f'Failed to write "{key}" data to shelve file')
                    print(ex)
                    continue


def load_py(file_path: Optional[Path] = None, variables: Optional[list[str]] = None,
            as_dict: bool = True, as_bulk: bool = False) \
        -> tuple[Optional[dict[str, Any]], Union[Path, str]]:
    if file_path and not re.search(r'\.[^\.]+$', str(file_path)):
        file_path = Path(f'{file_path}.dat')

    if (not file_path) or (not file_path.is_file()):
        file_path = filedialog.askopenfilename(defaultextension='.dat',
                                               filetypes=[('All Files', '*.*'), ('Shelve Files', '*.dat')],
                                               initialdir=Path.home(),
                                               title='Open a Python shelf:')

    if not file_path:
        return None, ''
    else:
        file_path = Path(file_path)
        file_path = Path(re.split(r'\.[^\.]+$', str(file_path))[0])

    data = {}
    with shelve.open(str(file_path), 'r') as shelve_file:
        for key, value in shelve_file.items():
            if variables:
                if key in variables:
                    if as_dict:
                        data[key] = value
                    else:
                        globals()[key] = value
            elif as_dict:
                data[key] = value
            else:
                globals()[key] = value

    if as_dict:
        if as_bulk:
            data = data['data']
        return data, file_path
    else:
        return None, file_path


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
    # save_py(test_file, data=_data)
    # load_py(test_file, variables=['str', 'lst', 'num', 'arr', 'dct'])

    # load_py(Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\05-May\23_0509\analysis\
    # Scan028\UC_VisaEBeam7\profiles_analysis.dat'))
    # noinspection PyUnboundLocalVariable
    # average_analysis = uns.UndulatorNoScan.is_image_valid(average_analysis, 2.)

    analysis, analysis_file = load_py(as_dict=True)

    print('done')
