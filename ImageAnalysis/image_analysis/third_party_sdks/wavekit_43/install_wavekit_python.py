import socket
import platform
from subprocess import Popen, PIPE
import os, sys
import importlib

this_file_path = os.path.dirname(os.path.abspath(__file__))

def list_files_in_dir(dir_to_list):
    onlyfiles = [f for f in os.listdir(dir_to_list) if os.path.isfile(os.path.join(dir_to_list, f))]
    return onlyfiles

def get_module_wheel(wheels_path, module_name):
    wheel_files = list_files_in_dir(wheels_path)
    for wheel_file in wheel_files:
        if(wheel_file[:len(module_name)].lower() == module_name.lower()):
            return wheel_file
    return -1

def get_pyton_short_version():
    return sys.version[:3]

def get_pyton_short_architecture():
    return platform.architecture()[0]

def is_connected_to_internet(url = 'one.one.one.one'):
    try:
        host = socket.gethostbyname(url)
        s = socket.create_connection((url, 80), 2)
        s.close()
        return True
    except:
        pass
    return False

def run(command):
    process = Popen(command, stdout=PIPE, shell=True)
    while True:
        line = process.stdout.readline().rstrip()
        if not line:
            break
        yield line
        

if __name__ == "__main__":
    python_path = os.path.dirname(os.path.abspath(sys.executable))
    print("python install path : " + python_path)
    
    needed_modules = ["numpy", "PyWin32"]
    needed_modules_imports = ["numpy", "win32api"]

    ###Install Needed Modules if not available
    print("----------------------")
    modules_to_install = []
    for i in range(len(needed_modules)) :
        try:
            importlib.import_module(needed_modules_imports[i])
        except:
            if(is_connected_to_internet()): modules_to_install.append(needed_modules[i])
            else:
                try:
                    wheels_path = os.path.join(this_file_path, 'wheels', get_pyton_short_version(), get_pyton_short_architecture())
                    wheel_file  = os.path.join(wheels_path, get_module_wheel(wheels_path, needed_modules[i]))
                    modules_to_install.append(wheel_file)
                except:
                    raise('INSTALL ERROR','No module available in wheel repository, please check your python version and download wheels file on PyPI')
                
    print("installing following modules : " + " ".join(modules_to_install))
    for module_to_install in modules_to_install:
        if(is_connected_to_internet()):
            cmd = '"{}"'.format(os.path.join(python_path, "python.exe")) + " -m pip install " + module_to_install
        else:
            cmd = '"{}"'.format(os.path.join(python_path, "python.exe")) + " -m pip install " + '"{}"'.format(module_to_install)
        print(cmd)
        for path in run(cmd):
            print (path)


    ###Add Module to Python Path
    print("----------------------")
    site_packages_path = os.path.join(python_path, 'Lib', 'site-packages')
    wavekit_python_path = os.path.join(site_packages_path, 'wavekit_python.pth')
    print("opening : " + wavekit_python_path)
    wavekit_python_pth_file = open(wavekit_python_path,'w')
    print("writing : " + this_file_path)
    wavekit_python_pth_file.write(this_file_path)
    writed_path = this_file_path
    
    if('WaveKit' in this_file_path):
        this_file_path.replace('WaveKit', 'WaveKitX64')
    elif('WaveKitX64' in this_file_path):
        this_file_path.replace('WaveKitX64', 'WaveKit')   
        
    if(os.path.exists(this_file_path) and writed_path != this_file_path):
        wavekit_python_pth_file.write('\n')  
        print("writing : " + this_file_path)
        wavekit_python_pth_file.write(this_file_path)       
    wavekit_python_pth_file.close()

    
    print("----------------------")
    print("Installation finished!")
    input('(Press Enter key to quit installer)')