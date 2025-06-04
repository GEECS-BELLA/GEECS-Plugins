from ctypes import cdll
from os import getcwd, path, chdir
import platform
from _ctypes import FreeLibrary
try: import winreg
except:
    try: 
        import _winreg 
        winreg = _winreg
    except: print('Can\'t load winreg module')

rVersion="4.3.1"

def load_dll() :
    mode = 'R'
    log = 'D'

    if(log == 'D'):
        log_f = open('log_dll_load.txt','a+')
        log_f.write('---------------------------------------------------------------------------\n')
    current_dir = getcwd()

    dlls_dir = path.join(path.dirname(path.realpath(__file__)), '..','dlls')
    if(winreg):
        if(not path.exists(dlls_dir)):
            rKey = 'SOFTWARE\\Imagine Optic\\RunTimeCore4\\'+rVersion+'\\FolderPath'
            rReg = winreg.ConnectRegistry(None,winreg.HKEY_LOCAL_MACHINE)
            dlls_dir = winreg.OpenKey(rReg, rKey)
    dll_name = 0
    if(mode == 'R'):
        if(platform.architecture()[0] == '32bit'):
            dlls_dir = path.join(dlls_dir, 'Win32')
            dll_name = 'imop_wavekit_4_c_vc141_Win32.dll'
        if(platform.architecture()[0] == '64bit'):
            dlls_dir = path.join(dlls_dir, 'x64')
            dll_name = 'imop_wavekit_4_c_vc141_x64.dll'
    elif(mode == 'D'):
        if(platform.architecture()[0] == '32bit'):
            dlls_dir = path.join(dlls_dir, 'Win32d')
            dll_name = 'imop_wavekit_4_c_vc141_Win32d.dll'
        if(platform.architecture()[0] == '64bit'):
            dlls_dir = path.join(dlls_dir, 'x64d')
            dll_name = 'imop_wavekit_4_c_vc141_x64d.dll'
                
    try:
        dll_full_file_name = path.join(dlls_dir, dll_name)
        chdir(dlls_dir)
        if(log == 'D'):
            log_f.write('dlls_dir           : ' + str(dlls_dir) + '\n')
            log_f.write('dll_name           : ' + str(dll_name) + '\n')
            log_f.write('dll_full_file_name : ' + str(dll_full_file_name) + '\n')
            log_f.write('cur_path           : ' + str(getcwd()) + '\n')
        dll = cdll.LoadLibrary(dll_full_file_name)
        chdir(current_dir)
        if(log == 'D'):
            log_f.close()
        return dll
    except Exception as e:
        if(log == 'D'):
            log_f.write('Exception : ' + str(e) + '\n\n')
            log_f.close()
        chdir(current_dir)
        raise Exception('IO_Error','---CAN NOT GET DLLS---')

def free_dll(handle) :
    FreeLibrary(handle)