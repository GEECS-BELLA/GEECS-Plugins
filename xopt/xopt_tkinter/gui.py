import csv
import tkinter as tk
from tkinter import ttk  # Import the themed tkinter module
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog

import threading


import yaml
import os
import time
import sys

class MyGUI(tk.Frame):
    def __init__(self, master=None, backend=None,geecs_interface=None, root_window=None,objective_function=None):
        super().__init__(master)
        self.master = master
        self.backend = backend
        self.geecs_interface=geecs_interface
        self.objective_function=objective_function
        self.root_window = root_window  # Store the root window
        self.controls = []  # List to keep track of added controls
        self.objectives = []  # List to keep track of added controls
        
        # Initialize the list for dynamically created controls for optimization method
        self.dynamic_controls = []
        self.current_row = 0  # Keep track of the current grid row
        
        # self.grid(sticky="nsew")  # Use grid() instead of pack() for finer control.
        
        # Initialize the attributes before calling create_widgets
        self.normalize_var = tk.BooleanVar()
        self.opt_target_device_var = tk.StringVar()
        self.opt_target_var_name_var = tk.StringVar()
        self.opt_steps_var = tk.IntVar()
        self.shots_per_step_var = tk.IntVar()
        self.opt_method_var = tk.StringVar()
        self.disable_sets_var = tk.BooleanVar()
        self.target_function_var = tk.StringVar()
        
        self.continue_updating = True
        
        
        self.objective_device_entries = {}
        
        self.create_widgets()

        # Set the protocol on the root_window, not the master
        self.root_window.protocol("WM_DELETE_WINDOW", self.on_closing)  # This line is changed
        
        # Configure the grid to expand. You might need to adjust the row and column numbers based on your layout.
        self.master.grid_rowconfigure(0, weight=1)  # Allows the first row to expand
        self.master.grid_columnconfigure(0, weight=1)  # Allows the first column to expand
        # If you have more rows or columns that need to expand, configure them similarly.
        
        self.default_values = {
            'normalize': True,
            'opt_method': 'bayes_ucb',
            'opt_steps': 20,
            'shots_per_step': 10,
            'opt_target_device': 'device',
            'opt_target_var_name': 'var',
            'disable_sets': True,
            'target_function':'None'
            
        }

        self.set_default_values()
        self.trace_vars()
        self.on_config_change()
        self.update_objective_variables()        
        
    def set_default_values(self):
        for key, value in self.default_values.items():
            getattr(self, f"{key}_var").set(value)
    
    def trace_vars(self):
        for key in self.default_values.keys():
            getattr(self, f"{key}_var").trace('w', self.on_config_change)

    def create_widgets(self):
        self.create_main_controls()
        self.create_objective_variables()  # Add this line
        self.create_config_controls()
        self.create_button_controls()
        self.add_control()
        self.add_obj_var()
        self.on_dropdown_change()
    
    def create_main_controls(self):
        self.main_controls_frame = tk.Frame(self.master)
        self.main_controls_frame.grid(row=0, column=0, padx=2, pady=5, sticky="nsew")
    

        # Only configure columns to expand, not the rows
        for col in range(7):  # You have 5 columns
            self.main_controls_frame.grid_columnconfigure(col, weight=1)

        labels = ["Device Name", "Variable Name", "Min", "Max", "Last Value","Use Current Pos","Delta"]
        for col, label_text in enumerate(labels):
            tk.Label(self.main_controls_frame, text=label_text).grid(row=0, column=col, sticky='ew')
    
    def create_objective_variables(self):
        self.objective_variables_frame = tk.Frame(self.master ,height=200)  # Added height
        self.objective_variables_frame.grid(row=1, column=0, padx=2, pady=5, sticky="nsew")
        self.objective_variables_frame.grid_propagate(False)  # Prevents the frame from shrinking beyond the specified height

        # Only configure columns to expand, not the rows
        for col in range(4):  # You have 4 columns
            self.objective_variables_frame.grid_columnconfigure(col, weight=1)

        labels = ["Device Name", "Variable Names", 'Key', "Last Value"]
        for col, label_text in enumerate(labels):
            tk.Label(self.objective_variables_frame, text=label_text).grid(row=0, column=col, sticky='ew')

    def create_config_controls(self):
        
        self.opt_method_options = ['bayes_UCB', 'nelder']
        config_entries = [
            ("Use Normalization", self.normalize_var, tk.BooleanVar, tk.Checkbutton, None),
            ("Method", self.opt_method_var, tk.StringVar, ttk.Combobox, 15),
            ("Optimization Steps:", self.opt_steps_var, tk.IntVar, tk.Entry, 10),
            ("Shots per step:", self.shots_per_step_var, tk.IntVar, tk.Entry, 10),
            # ("Objective Device:", self.opt_target_device_var, tk.StringVar, tk.Entry, 15),
            # ("Objective Var:", self.opt_target_var_name_var, tk.StringVar, tk.Entry, 15),
            ("Disable sets", self.disable_sets_var, tk.BooleanVar, tk.Checkbutton, None),
            ('Target function: ',self.target_function_var, tk.StringVar, tk.Entry,15)
        ]

    
        desired_max_width, desired_max_height = 400, 200
        self.config_frame = tk.Frame(self.master)
        self.config_frame.grid(row=0, column=4, padx=10, pady=5)
        self.config_frame.grid_propagate(False)
        self.config_frame.config(width=desired_max_width, height=desired_max_height)

        for row, (label_text, var, var_type, widget_type, width) in enumerate(config_entries):
            tk.Label(self.config_frame, text=label_text).grid(row=row, column=0)
            if widget_type == ttk.Combobox:
                widget = widget_type(self.config_frame, textvariable=var, values=self.opt_method_options, width=width)
            else:
                widget_args = {"textvariable": var} if var_type != tk.BooleanVar else {"variable": var}
                if width:
                    widget_args["width"] = width
                widget = widget_type(self.config_frame, **widget_args)
            widget.grid(row=row, column=1)
    
    def create_button_controls(self):
        self.button_controls_frame = tk.Frame(self.master)
        self.button_controls_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

        button_data_row_0 = [
            ("Add Control", self.add_control),
            ("Remove Control", self.remove_control),
            # ("Load Controls Config", self.load_controls_config_button),
            ("Load Master Config", self.load_master_config)
        ]
        button_data_row_1 = [
            ("Add Obj", self.add_obj_var),
            ("Remove Obj", self.remove_obj_var),
            ("Intialize Devices", self.intialize_geecs_devices)
            
        ]
        for col, (btn_text, cmd) in enumerate(button_data_row_0):
            tk.Button(self.button_controls_frame, text=btn_text, command=cmd).grid(row=0, column=col, sticky='ew')

        for col, (btn_text, cmd) in enumerate(button_data_row_1):
            tk.Button(self.button_controls_frame, text=btn_text, command=cmd).grid(row=1, column=col, sticky='ew')
                      
    def load_master_config(self):
        filepath = self.get_filepath("YAML", "*.yaml", "Select a Config File")
        if not filepath:
            return
        master_config = self.load_yaml(filepath)
        self.set_from_yaml(master_config, 'controls_file', self.load_controls_config)
        # Load objective variables
        if 'objective_targets' in master_config and 'objective_function' in master_config:
            if not filepath:
                return

        elif 'objective_targets' in master_config:
            # Update the objectives in geecs_interface
            self.geecs_interface.objectives_dict = master_config['objective_targets']

            # Load the objectives into the GUI
            self.load_objectives_config(master_config['objective_targets'])

            class_name = 'default'
            master_config['objective_function']=class_name
            ObjectiveClass = self.geecs_interface.get_objective_function_class(class_name=class_name)
            obj_instance = ObjectiveClass()
            self.geecs_interface.obj_instance = obj_instance
            self.geecs_interface.obj_instance.variables = master_config['objective_targets']


        elif 'objective_function' in master_config:
            class_name = master_config['objective_function']
            ObjectiveClass = self.geecs_interface.get_objective_function_class(class_name=class_name)
            obj_instance = ObjectiveClass()
            self.geecs_interface.objectives_dict = obj_instance.variables
            self.geecs_interface.obj_instance = obj_instance

            # Load the objectives into the GUI
            self.load_objectives_config(self.geecs_interface.objectives_dict)

        self.set_from_yaml(master_config, 'optimization_steps', self.opt_steps_var.set, 20)
        self.set_from_yaml(master_config, 'shots_per_step', self.shots_per_step_var.set, 20)
        self.set_from_yaml(master_config, 'disable_sets', self.disable_sets_var.set, True)
        self.set_from_yaml(master_config, 'objective_function', self.target_function_var.set, "None")
        
    def get_yaml_files(self):
        config_dir = 'config_files'
        return [f for f in os.listdir(config_dir) if f.endswith('.yaml') or f.endswith('.yml')]
    
    def get_filepath(self, file_type, file_pattern, title):
        yaml_files = self.get_yaml_files()
        if not yaml_files:
            messagebox.showinfo("Info", "No YAML files found in config_files directory.")
            return

        # Create a top-level window for file selection
        top = tk.Toplevel(self.master)
        top.title(title)

        # Label and Combobox for file selection
        tk.Label(top, text="Select a Config File:").pack(pady=10)
        file_var = tk.StringVar(top)
        combo = ttk.Combobox(top, textvariable=file_var, values=yaml_files)
        combo.pack(pady=10, padx=10)

        # Function to set the selected file and destroy the top-level window
        def on_select():
            self.selected_file = file_var.get()
            top.destroy()

        # OK button to confirm the selection
        tk.Button(top, text="OK", command=on_select).pack(pady=10)

        # This line makes the method wait until the top-level window is closed
        self.master.wait_window(top)

        return os.path.join('config_files', self.selected_file) if hasattr(self, 'selected_file') else None
            
    def add_entry(self, parent_frame, row, col, width, initial_val=""):
        entry = tk.Entry(parent_frame, width=width)
        entry.grid(row=row, column=col)
        entry.insert(0, initial_val)
        return entry

    def add_generic_control(self, parent_frame, control_list, control_specs, *initial_vals):
        # next_row = len(control_list) + 1  # Calculate the row for the new controls
        #
        # entries = []
        # for col, (width, initial_val) in enumerate(zip(control_specs, initial_vals)):
        #     entry = self.add_entry(parent_frame, next_row, col, width, initial_val)
        #     entries.append(entry)
        #
        # control_list.append(tuple(entries))
        
        next_row = len(control_list) + 1  # Calculate the row for the new controls

        entries = []
        for col, (width, initial_val) in enumerate(zip(control_specs, initial_vals)):
            if col == 5:  # Assuming "Use Current Pos" is at column 5
                var = tk.BooleanVar(value=initial_val)
                entry = tk.Checkbutton(parent_frame, variable=var)
                entry.grid(row=next_row, column=col, sticky='ew')
                entry.var = var  # Store the variable for later retrieval
            else:
                entry = self.add_entry(parent_frame, next_row, col, width, initial_val)
            entries.append(entry)

        control_list.append(tuple(entries))
        
    def remove_generic_control(self, control_list):
        if control_list:
            last_control_set = control_list.pop()
            for widget in last_control_set:
                widget.grid_forget()
                widget.destroy()

    def add_control(self, device_name="", variable_name="", min_val="", max_val="", last_value="", use_current_pos=False, delta=""):
        control_specs = [15, 10, 5, 5, 10, 5, 5]  # widths for each entry
        # self.add_generic_control(self.main_controls_frame, self.controls, control_specs, device_name, variable_name, min_val, max_val, last_value,use_current_pos,delta)
        self.add_generic_control(self.main_controls_frame, self.controls, control_specs, device_name, variable_name, min_val, max_val, last_value, use_current_pos, delta)
         
    def add_obj_var(self, device_name="", variable_name="", key="", last_value=""):
        control_specs = [15, 20, 5, 15]  # widths for each entry
        self.add_generic_control(self.objective_variables_frame, self.objectives, control_specs, device_name, variable_name, key, last_value)
        
        # Assuming the device_name entry is the first in the tuple, store a reference to it
        last_value_entry = self.objectives[-1][3]
        self.objective_device_entries[key] = last_value_entry
        
    def remove_control(self):
        self.remove_generic_control(self.controls)

    def remove_obj_var(self):
        self.remove_generic_control(self.objectives)
    
    def load_yaml(self, filepath):
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)
    
    def set_from_yaml(self, yaml_data, key, setter_func, default=None):
        if key in yaml_data:
            setter_func(yaml_data[key])
        elif default:
            setter_func(default)
        
    def load_controls_config(self,filename):
        # Clear all current controls
        while self.controls:
            last_control_set = self.controls.pop()
            for widget in last_control_set:
                widget.grid_forget()
                widget.destroy()
        
        try:
            directory='config_files'
            filepath = os.path.join(directory, filename)
            # Read and parse the TSV file
            with open(filepath, newline='') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')  # Specify tab delimiter
                next(reader, None)  # Skip the header row
                for row in reader:
                    if len(row) == 6:  # Check if the row contains the right number of columns
                        device_name, variable_name, min_val, max_val, use_current_pos, delta = row    
                        self.add_control(device_name=device_name, variable_name=variable_name, min_val=min_val, max_val=max_val,
                        last_value="NA", use_current_pos=use_current_pos, delta=delta)
        except Exception as e:
            tk.messagebox.showerror("Error loading file", str(e)) 
            
    def load_objectives_config(self, objectives_dict):
        # Clear all current objectives
        while self.objectives:
            last_objective_set = self.objectives.pop()
            for widget in last_objective_set:
                widget.grid_forget()
                widget.destroy()

        try:
            for key, obj_info in objectives_dict.items():
                device_name = obj_info['device_name']
                # Convert the list of device_subscribe_variables into a comma-separated string
                variable_names_str = ', '.join(obj_info['device_subscribe_variables'])
                # Add to the GUI with the key name
                self.add_obj_var(key=key, device_name=device_name, variable_name=variable_names_str, last_value="NA")
        except Exception as e:
            tk.messagebox.showerror("Error loading objectives", str(e))
                
    def load_controls_config_button(self):
        # Open a file dialog to select the TSV file
        filepath = filedialog.askopenfilename(
            filetypes=[("TSV Files", "*.tsv"), ("Text Files", "*.txt")], title="Select a Config File"
        )
        if not filepath:  # Exit the function if no file was selected
            return

        self.load_controls_config(filepath)
        
    def on_config_change(self, *args):
        new_params = {
            'normalize': self.normalize_var.get(),
            'opt_method': self.opt_method_var.get(),
            'opt_steps':  self.opt_steps_var.get(),
            'shots_per_step': self.shots_per_step_var.get(),
            # 'opt_target_device': self.opt_target_device_var.get(),
            # 'opt_target_var_name':self.opt_target_var_name_var.get(),
            'disable_sets':self.disable_sets_var.get()
        }
        self.backend.set_config_params(new_params)
        self.backend.configure_yaml(self.geecs_interface.backend_vocs)
    
    def on_dropdown_change(self, *args):
        
        # Remove any existing dynamic controls
        for widget in self.dynamic_controls:
            widget.grid_forget()
            widget.destroy()

        # Clear the list
        self.dynamic_controls = []
        
        # Get the current selected method from the dropdown
        selected_method = self.opt_method_var.get()

        # # Check which method is selected and decide which input fields to show
        # if selected_method == 'bayes_UCB':
        #     self.show_bayes_UCB_input_fields()
        # elif selected_method == 'nelder':
        #     self.show_nelder_input_fields()
        # else:
        #     self.hide_all_input_fields()
            
    def show_bayes_UCB_input_fields(self):
        # Create/show the input fields related to bayes_UCB
        # For example:
        self.bayes_UCB_label = tk.Label(self.config_frame, text="n_initial:")
        self.bayes_UCB_label.grid(row=6, column=0)
        self.bayes_UCB_entry = tk.Entry(self.config_frame)
        self.bayes_UCB_entry.grid(row=6, column=1)
        self.bayes_UCB_label2 = tk.Label(self.config_frame, text="acq:")
        self.bayes_UCB_label2.grid(row=7, column=0)
        self.bayes_UCB_entry2 = tk.Entry(self.config_frame)
        self.bayes_UCB_entry2.grid(row=7, column=1)
        self.dynamic_controls.extend([self.bayes_UCB_label, self.bayes_UCB_entry, self.bayes_UCB_label2, self.bayes_UCB_entry2])        

    def show_nelder_input_fields(self):
        # Create/show the input fields related to nelder
        # For example:
        self.nelder_label = tk.Label(self.config_frame, text="Nelder Parameter:")
        self.nelder_label.grid(row=6, column=0)
        self.nelder_entry = tk.Entry(self.config_frame)
        self.nelder_entry.grid(row=6, column=1)
        self.dynamic_controls.extend([self.nelder_label, self.nelder_entry])     

    def hide_all_input_fields(self):
        # Hide all additional input fields if any exist
        for widget in [getattr(self, attr) for attr in dir(self) if 'label' in attr or 'entry' in attr]:
            widget.grid_forget()
    
    def intialize_geecs_devices(self):
        if not self.geecs_interface.devices_active:
            self.initialize_controls()
            self.initialize_objective()
        else:
            self.geecs_interface.close_all_controls()
            self.geecs_interface.close_all_objectives()
            time.sleep(1)
            self.initialize_controls()
            self.initialize_objective()

    def initialize_objective(self):
        self.geecs_interface.objectives_dict #should be updated here
        self.geecs_interface.initialize_objective()   

    def initialize_controls(self):
        # Create a list to hold the data from all controls
        all_controls_data = []

        # Retrieve the text from all the controls' input fields
        for device_entry, variable_entry, min_entry, max_entry, last_value_entry, use_current_pos_entry, delta_entry in self.controls:
            control_data = {
                "device_name": device_entry.get(),
                "variable_name": variable_entry.get(),
                "min_value": min_entry.get(),
                "max_value": max_entry.get(),
                "last_value": last_value_entry.get(),
                "use_current_pos":use_current_pos_entry.var.get(),
                "delta":delta_entry.get()
            }
            all_controls_data.append(control_data)

        # Pass the list of control data to the backend for processing
        initialization_results = self.geecs_interface.initialize_all_controls(all_controls_data)
        
        # After performing the initializations, retrieve the last acquired values for each control
        for control_data, control_widgets in zip(all_controls_data, self.controls):
            device_name = control_data['device_name']
            variable_name = control_data['variable_name']
            key = f"{device_name}::{variable_name}"

            # Assuming your backend has a method get_last_acquired_value that takes a device name and variable name
            last_value = self.geecs_interface.get_last_acquired_value(device_name, variable_name)
            last_value_entry = control_widgets[4]
            last_value_entry.delete(0, tk.END)  # Clear the current text
            last_value_entry.insert(0, str(last_value))  # Insert the new text
            
            bounds = self.geecs_interface.devices[key]['bounds']
            min_entry = control_widgets[2]
            min_entry.delete(0, tk.END)  # Clear the current text
            min_entry.insert(0, str(bounds[0]))  # Insert the new text
            
            max_entry = control_widgets[3]
            max_entry.delete(0, tk.END)  # Clear the current text
            max_entry.insert(0, str(bounds[1]))  # Insert the new text
             
    def update_objective_variables(self):
        current_values = self.geecs_interface.get_obj_var_tcp_state(wait_for_new=False)

        for device_key, metrics in current_values.items():
            # Extract all metric values and join them with commas
            values_as_str = ", ".join(str(value) for value in metrics.values())

            # Update the corresponding entry in the GUI
            if device_key in self.objective_device_entries:
                entry_widget_for_device = self.objective_device_entries[device_key]
                entry_widget_for_device.delete(0, tk.END)
                entry_widget_for_device.insert(0, values_as_str)

        if self.continue_updating:
            self.after(1000, self.update_objective_variables)
        
    def stop_updates(self):
        self.continue_updating = False
         
    def on_closing(self):
        self.stop_updates()
        
        # Custom close behavior: destroy the main window and break the mainloop
        self.geecs_interface.close_all_controls()
        self.geecs_interface.close_all_objectives()        
        self.root_window.destroy()  # Use the root_window attribute here
        self.root_window.quit()
