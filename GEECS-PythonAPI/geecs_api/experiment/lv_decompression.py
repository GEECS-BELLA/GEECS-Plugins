import os.path
import sys
from ctypes import *


if __name__ == '__main__':
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LVx32Compression.dll')
    print(dll_path)
    lv_dll = cdll.LoadLibrary(dll_path)
