import csv
import tkinter as tk
from tkinter import ttk  # Import the themed tkinter module
from tkinter import filedialog
import yaml

class MyGUI(tk.Frame):
    def __init__(self, master=None, backend=None,geecs_interface=None, root_window=None):
        super().__init__(master)
        self.master = master
        self.backend = backend
        self.geecs_interface=geecs_interface
        self.root_window = root_window  # Store the root window
        self.controls = []  # List to keep track of added controls
        # Initialize the list for dynamically created controls for optimization method
        self.dynamic_controls = []
        self.current_row = 0  # Keep track of the current grid row

        self.grid(sticky="nsew")  # Use grid() instead of pack() for finer control.
        
        # Initialize the attributes before calling create_widgets
        self.normalize_var = tk.BooleanVar()
        self.opt_target_device_var = tk.StringVar()
        self.opt_target_var_name_var = tk.StringVar()
        self.opt_steps_var = tk.IntVar()
        self.shots_per_step_var = tk.IntVar()
        self.opt_method_var = tk.StringVar()
        
        self.create_widgets()

        # Set the protocol on the root_window, not the master
        self.root_window.protocol("WM_DELETE_WINDOW", self.on_closing)  # This line is changed
        
        # Configure the grid to expand. You might need to adjust the row and column numbers based on your layout.
        self.master.grid_rowconfigure(0, weight=1)  # Allows the first row to expand
        self.master.grid_columnconfigure(0, weight=1)  # Allows the first column to expand
        # If you have more rows or columns that need to expand, configure them similarly.
        
        self.default_values = {
            'normalize': True,
            'opt_method': 'bayes_UCB',
            'opt_steps': 20,
            'shots_per_step': 10,
            'opt_target_device': 'device',
            'opt_target_var_name': 'var'
        }
        self.trace_vars()
        self.set_default_values()
        self.on_config_change()
        
    def set_default_values(self):
        for key, value in self.default_values.items():
            getattr(self, f"{key}_var").set(value)
    
    def trace_vars(self):
        for key in self.default_values.keys():
            getattr(self, f"{key}_var").trace('w', self.on_config_change)


    def create_widgets(self):
        self.create_main_controls()
        self.create_config_controls()
        self.create_button_controls()
        self.add_control()
        self.on_dropdown_change()

    def create_main_controls(self):
        labels = ["Device Name", "Variable Name", "Min", "Max", "Last Acquired Value"]
        for col, label_text in enumerate(labels):
            tk.Label(self, text=label_text).grid(row=0, column=col, sticky='ew')

    def create_config_controls(self):
        
        self.opt_method_options = ['bayes_UCB', 'nelder']
        config_entries = [
            ("Use Normalization", self.normalize_var, tk.BooleanVar, tk.Checkbutton, None),
            ("Method", self.opt_method_var, tk.StringVar, ttk.Combobox, 15),
            ("Optimization Steps:", self.opt_steps_var, tk.IntVar, tk.Entry, 10),
            ("Shots per step:", self.shots_per_step_var, tk.IntVar, tk.Entry, 10),
            ("Objective Device:", self.opt_target_device_var, tk.StringVar, tk.Entry, 15),
            ("Objective Var:", self.opt_target_var_name_var, tk.StringVar, tk.Entry, 15)
        ]

    
        desired_max_width, desired_max_height = 300, 200
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
        self.button_frame = tk.Frame(self.master)
        self.button_frame.grid(sticky="ew", row=1000, column=0, columnspan=4)

        button_data = [
            ("Add Control", self.add_control),
            ("Remove Control", self.remove_control),
            ("Load Controls Config", self.load_controls_config_button),
            ("Load Master Config", self.load_master_config),
            ("Intialize Devices", self.intialize_geecs_devices)
        ]
        for col, (btn_text, cmd) in enumerate(button_data):
            tk.Button(self.button_frame, text=btn_text, command=cmd).grid(row=0, column=col)

    def add_control(self, device_name="", variable_name="", min_val="", max_val="", last_value=""):
        next_row = len(self.controls) + 2  # Calculate the row for the new controls

        # Create a new set of inputs for the device name
        device_entry = tk.Entry(self, width=15)
        device_entry.grid(row=next_row, column=0)
        device_entry.insert(0, device_name)  # Insert the data if provided

        # Create a new set of inputs for the variable name
        variable_entry = tk.Entry(self, width=15)
        variable_entry.grid(row=next_row, column=1)
        variable_entry.insert(0, variable_name)  # Insert the data if provided

        # Create a new set of inputs for the minimum value
        min_entry = tk.Entry(self, width=7)
        min_entry.grid(row=next_row, column=2)
        min_entry.insert(0, min_val)  # Insert the data if provided

        # Create a new set of inputs for the maximum value
        max_entry = tk.Entry(self, width=7)
        max_entry.grid(row=next_row, column=3)
        max_entry.insert(0, max_val)  # Insert the data if provided

        # Create a new label for displaying the last acquired value
        last_value_label = tk.Entry(self, width=12)
        last_value_label.grid(row=next_row, column=4)
        last_value_label.insert(0, last_value)  # Insert the data if provided

        # Store these controls in the list
        self.controls.append((device_entry, variable_entry, min_entry, max_entry, last_value_label))
        

    def remove_control(self):
        if self.controls:  # Check if there are any controls to remove
            # Retrieve the last set of controls
            last_control_set = self.controls.pop()

            # Remove the last set of controls from the grid and delete them
            for widget in last_control_set:
                widget.grid_forget()
                widget.destroy()
                
    def load_master_config(self):
        filepath = self.get_filepath("YAML", "*.yaml", "Select a Config File")
        if not filepath:
            return
        master_config = self.load_yaml(filepath)
        self.set_from_yaml(master_config, 'controls_file', self.load_controls_config)
        self.set_from_yaml(master_config['objective_target'], 'device_name', self.opt_target_device_var.set, 'default_device_name')
        self.set_from_yaml(master_config['objective_target'], 'device_variable', self.opt_target_var_name_var.set, 'default_variable_name')
        self.set_from_yaml(master_config, 'optimization_steps', self.opt_steps_var.set, 20)
        self.set_from_yaml(master_config, 'shots_per_step', self.shots_per_step_var.set, 20)

    def get_filepath(self, file_type, file_pattern, title):
        return filedialog.askopenfilename(filetypes=[(file_type, file_pattern)], title=title)
    
    def load_yaml(self, filepath):
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)
    
    def set_from_yaml(self, yaml_data, key, setter_func, default=None):
        if key in yaml_data:
            setter_func(yaml_data[key])
        elif default:
            setter_func(default)
        
    def load_controls_config(self,filepath):
        # Clear all current controls
        while self.controls:
            last_control_set = self.controls.pop()
            for widget in last_control_set:
                widget.grid_forget()
                widget.destroy()
        
        try:
            # Read and parse the TSV file
            with open(filepath, newline='') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')  # Specify tab delimiter
                next(reader, None)  # Skip the header row
                for row in reader:
                    if len(row) == 4:  # Check if the row contains the right number of columns
                        device_name, variable_name, min_val, max_val = row
                        self.add_control(device_name=device_name, variable_name=variable_name, min_val=min_val, max_val=max_val,last_value="NA")
        except Exception as e:
            tk.messagebox.showerror("Error loading file", str(e))      

                
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
            'opt_target_device': self.opt_target_device_var.get(),
            'opt_target_var_name':self.opt_target_var_name_var.get()
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
        self.initialize_controls()
        self.initialize_objective()
    
    def initialize_objective(self):
        self.geecs_interface.initialize_objective(
            self.opt_target_device_var.get(), 
            self.opt_target_var_name_var.get()
        )

    def initialize_controls(self):
        # Create a list to hold the data from all controls
        all_controls_data = []

        # Retrieve the text from all the controls' input fields
        for device_entry, variable_entry, min_entry, max_entry, last_value_entry in self.controls:
            control_data = {
                "device_name": device_entry.get(),
                "variable_name": variable_entry.get(),
                "min_value": min_entry.get(),
                "max_value": max_entry.get(),
                "last_value": last_value_entry.get()
            }
            all_controls_data.append(control_data)

        # Pass the list of control data to the backend for processing
        initialization_results = self.geecs_interface.initialize_all_controls(all_controls_data)
        
        # After performing the initializations, retrieve the last acquired values for each control
        for control_data, control_widgets in zip(all_controls_data, self.controls):
            device_name = control_data['device_name']
            variable_name = control_data['variable_name']

            # Assuming your backend has a method get_last_acquired_value that takes a device name and variable name
            last_value = self.geecs_interface.get_last_acquired_value(device_name, variable_name)
            print(last_value)
            last_value_entry = control_widgets[4]
            last_value_entry.delete(0, tk.END)  # Clear the current text
            last_value_entry.insert(0, str(last_value))  # Insert the new text
        
    def on_closing(self):
        # Custom close behavior: destroy the main window and break the mainloop
        self.root_window.destroy()  # Use the root_window attribute here
        self.root_window.quit()





