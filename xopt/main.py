import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
import sys
import time

from xopt_tkinter.gui import MyGUI
from xopt_tkinter.optimization_tab import OptimizationTab

# from pyqt.gui import MyGUI
# from pyqt.optimization_tab import OptimizationTab

from backend import MyBackend
from geecs_functions import GeecsXoptInterface
# from geecs_functions import HTU_test_opt


def select_experiment():
    experiments = ['Undulator', 'BELLA', 'HTT']  # Add all your experiments to this list

    # Function to destroy the window
    def on_ok():
        window.destroy()

    window = tk.Tk()
    window.title("Select Experiment")
    
    experiment_var = tk.StringVar(window)
    experiment_var.set(experiments[0])  # set the default value

    dropdown = ttk.OptionMenu(window, experiment_var, *experiments)
    dropdown.pack(pady=20, padx=20)

    ok_button = ttk.Button(window, text="OK", command=on_ok)
    ok_button.pack(pady=20)

    window.mainloop()

    return experiment_var.get()

def main(experiment='Undulator'):
    root = tk.Tk()
    backend = MyBackend()
    geecs_interface = GeecsXoptInterface(experiment_name=prompted_name)

    # Create the tab control
    tabControl = ttk.Notebook(root)

    # Create the tab frames
    tab1 = ttk.Frame(tabControl)
    tab2 = ttk.Frame(tabControl)

    # Add the tabs to the tab control
    tabControl.add(tab1, text='Setup Controls')
    # Pass the root window to MyGUI
    app = MyGUI(master=tab1, backend=backend, geecs_interface=geecs_interface, root_window=root)

    tabControl.add(tab2, text='Optimization Terms')
    # Create an instance of OptimizationTab and place it in tab2
    optimization_app = OptimizationTab(master=tab2, backend=backend, geecs_interface=geecs_interface)
    optimization_app.pack(fill="both", expand=True)  # You need to pack the OptimizationTab frame here

    # Pack the tab control to make it visible
    tabControl.pack(expand=1, fill="both")

    root.mainloop()

if __name__ == "__main__":
    
    # Get the experiment name from the user
    prompted_name = select_experiment()
    
    import numpy as np
    def geecs_measurement(input_dict, normalize=None,shots_per_step=None,disable_sets=False):
        print(input_dict)
        print('shots_per_step',shots_per_step)
        geecs_interface = GeecsXoptInterface(prompted_name)

        print('disable_sets',disable_sets)
        
        for i in list(input_dict.keys()):
            try:
                set_val = float(input_dict[i])

                if normalize:
                    set_val = geecs_interface.unnormalize_controls(i, set_val)

                print('set ' + str(i) + ' to ' + str(set_val))
                print(geecs_interface.devices[i])
                
                if disable_sets:
                    print('setting of controls turned off')
                else:
                    print('setting of controls turned on')
                    geecs_interface.devices[i]["GEECS_Object"].set(geecs_interface.devices[i]["variable"], set_val)
                time.sleep(0)

            except Exception as e:
                print(f"An error occurred: {e}")
                break

        if normalize:
            setpoint = {key: geecs_interface.unnormalize_controls(key, input_dict[key]) for key in input_dict}
        else:
            setpoint = input_dict

        print(setpoint)
        
        values=[]
        for i in range(shots_per_step):
            print('shot num', i)
            measured_values=geecs_interface.get_obj_var_tcp_state()
            value=geecs_interface.obj_instance.calculate(measured_values)
            # value = geecs_interface.calcTransmission(setpoint)
            print(value)
            values.append(value)
            
        result=np.array(values)

        return {'f': result}

    main(experiment=prompted_name)
    
    


