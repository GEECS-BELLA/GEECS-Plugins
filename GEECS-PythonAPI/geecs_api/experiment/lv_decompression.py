import os.path
import sys
import ctypes as ct


if __name__ == '__main__':
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lv32add', 'lv32add.dll')
    print(dll_path)
    # ct.windll.LoadLibrary('lv32add.dll')
    ct.windll.LoadLibrary(dll_path)
