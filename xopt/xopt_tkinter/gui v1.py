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
        
        # Static labels at the top of the application, created before adding controls
        tk.Label(self, text="Device Name").grid(row=0, column=0, sticky='ew')
        tk.Label(self, text="Variable Name").grid(row=0, column=1, sticky='ew')
        tk.Label(self, text="Min").grid(row=0, column=2, sticky='ew')
        tk.Label(self, text="Max").grid(row=0, column=3, sticky='ew')
        tk.Label(self, text="Last Acquired Value").grid(row=0, column=4, sticky='ew')

        self.grid(sticky="nsew")  # Use grid() instead of pack() for finer control.
        
        self.create_widgets()

        # Set the protocol on the root_window, not the master
        self.root_window.protocol("WM_DELETE_WINDOW", self.on_closing)  # This line is changed
        
        # Configure the grid to expand. You might need to adjust the row and column numbers based on your layout.
        self.master.grid_rowconfigure(0, weight=1)  # Allows the first row to expand
        self.master.grid_columnconfigure(0, weight=1)  # Allows the first column to expand
        # If you have more rows or columns that need to expand, configure them similarly.
        
        # Set default values
        default_normalize = True  # or False, depending on your preference
        default_opt_method = 'bayes_UCB'  # or 'nelder', depending on your preference  
        default_opt_steps = 20  
        default_shots_per_step = 10
        default_opt_target_device = 'device'
        default_opt_target_var_name = 'var'
        
        self.normalize_var.set(default_normalize)
        self.opt_method_var.set(default_opt_method)
        self.opt_steps_var.set(default_opt_steps)
        self.shots_per_step_var.set(default_shots_per_step)
        self.opt_target_device_var.set(default_opt_target_device)
        self.opt_target_var_name_var.set(default_opt_target_var_name)
        
        

        self.normalize_var.trace('w', self.on_config_change)
        self.opt_method_var.trace('w', self.on_config_change)        
        self.shots_per_step_var.trace('w', self.on_config_change)
        self.opt_steps_var.trace('w', self.on_config_change)
        self.opt_target_device_var.trace('w', self.on_config_change)
        self.opt_target_var_name_var.trace('w', self.on_config_change)
        
        # Initialize the backend with the default settings
        self.on_config_change()
        



    def create_widgets(self):
        # Create a frame for the buttons, which will be positioned at the bottom
        self.button_frame = tk.Frame(self.master)
        self.button_frame.grid(sticky="ew", row=1000, column=0, columnspan=4)  # Placed at the bottom by using a high row number

        # Create the "Add Control" button inside the button frame
        self.add_control_button = tk.Button(self.button_frame, text="Add Control", command=self.add_control)
        self.add_control_button.grid(row=0, column=0)

        # Create the "Remove Control" button inside the button frame
        self.remove_control_button = tk.Button(self.button_frame, text="Remove Control", command=self.remove_control)
        self.remove_control_button.grid(row=0, column=1)
        
        # Add a new "Load Config" button inside the button frame
        self.load_config_button = tk.Button(self.button_frame, text="Load Controls Config", command=self.load_controls_config_button)
        self.load_config_button.grid(row=0, column=2)  # place it next to the other buttons
        
        # Add a new "Load Config" button inside the button frame
        self.load_config_button = tk.Button(self.button_frame, text="Load Master Config", command=self.load_master_config)
        self.load_config_button.grid(row=0, column=3)  # place it next to the other buttons
        
        # Create the "Intialize Controls" button inside the button frame
        self.some_button = tk.Button(self.button_frame, text="Intialize Devices", command=self.initialize_controls)
        self.some_button.grid(row=0, column=4)
        
        # Create a new frame for the configuration settings
        desired_max_width, desired_max_height = 300,200
        self.config_frame = tk.Frame(self.master)
        self.config_frame.grid(row=0, column=4, padx=10, pady=5)  # Adjust grid column and padding as necessary
        self.config_frame.grid_propagate(False)  # Prevents children widgets from resizing the frame
        self.config_frame.config(width=desired_max_width, height=desired_max_height)
        

        # Checkbox for normalization inside the config frame
        self.normalize_var = tk.BooleanVar()
        self.normalize_check = tk.Checkbutton(self.config_frame, text="Use Normalization", variable=self.normalize_var)
        self.normalize_check.grid(row=0, column=0, sticky="w")  # Adjust grid as necessary

        # Dropdown for optimization method inside the config frame
        self.opt_method_var = tk.StringVar()
        self.opt_method_options = ['bayes_UCB', 'nelder']
        self.opt_method_menu = ttk.Combobox(self.config_frame, textvariable=self.opt_method_var, values=self.opt_method_options, state='readonly',width=10)
        self.opt_method_menu.grid(row=1, column=0, sticky="ew")  # Adjust grid as necessary
        self.opt_method_var.set(self.opt_method_options[0])  # default value
        self.opt_method_var.trace('w', self.on_dropdown_change)
        
        self.opt_steps_var = tk.IntVar()
        self.opt_steps_label = tk.Label(self.config_frame, text="optimization steps:")
        self.opt_steps_label.grid(row=2, column=0)
        self.opt_steps_entry = tk.Entry(self.config_frame, textvariable=self.opt_steps_var)
        self.opt_steps_entry.grid(row=2, column=1)
        
        self.shots_per_step_var = tk.IntVar()
        self.shots_per_step_label = tk.Label(self.config_frame, text="shots per step:")
        self.shots_per_step_label.grid(row=3, column=0)
        self.shots_per_step_entry = tk.Entry(self.config_frame, textvariable=self.shots_per_step_var)
        self.shots_per_step_entry.grid(row=3, column=1)
        
        self.opt_target_device_var = tk.StringVar()
        self.opt_target_device_label = tk.Label(self.config_frame, text="Objective Device:")
        self.opt_target_device_label.grid(row=4, column=0)
        self.opt_target_device_entry = tk.Entry(self.config_frame, textvariable=self.opt_target_device_var)
        self.opt_target_device_entry.grid(row=4, column=1)
        
        self.opt_target_var_name_var = tk.StringVar()
        self.opt_target_var_name_label = tk.Label(self.config_frame, text="Objective Var:")
        self.opt_target_var_name_label.grid(row=5, column=0)
        self.opt_target_var_name_entry = tk.Entry(self.config_frame, textvariable=self.opt_target_var_name_var)
        self.opt_target_var_name_entry.grid(row=5, column=1)
        
        # Initialize the GUI with one set of control inputs
        self.add_control()
        self.on_dropdown_change()
        

    def add_control(self):
        # Calculate the row where the new controls should be added. 
        # This should be after the existing controls and the row with static labels.
        next_row = len(self.controls) + 2  # We start from 2 because row 0 has static labels and row 1 will have the first control

        # Create a new set of inputs for the device name
        device_entry = tk.Entry(self,width=15)
        device_entry.grid(row=next_row, column=0)

        # Create a new set of inputs for the variable name
        variable_entry = tk.Entry(self,width=15)
        variable_entry.grid(row=next_row, column=1)

        # Create a new set of inputs for the minimum value
        min_entry = tk.Entry(self,width=7)
        min_entry.grid(row=next_row, column=2)

        # Create a new set of inputs for the maximum value
        max_entry = tk.Entry(self,width=7)
        max_entry.grid(row=next_row, column=3)
        
        # Create a new label for displaying the last acquired value, which starts empty
        last_value_label = tk.Entry(self,width=12)
        last_value_label.grid(row=next_row, column=4)

        # Store these controls in the list so we can retrieve the data later or remove controls if needed
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
        # Open a file dialog to select the TSV file
        filepath = filedialog.askopenfilename(
            filetypes=[("YAML", "*.yaml")], title="Select a Config File"
        )
        if not filepath:  # Exit the function if no file was selected
            return
            
        # Load the YAML content from the provided file
        with open(filepath, 'r') as file:
           master_config = yaml.safe_load(file)
               
        if 'controls_file' in master_config:
            self.load_controls_config(master_config['controls_file'])

        if 'objective_target' in master_config:
            self.opt_target_device_var.set(master_config['objective_target'].get('device_name', 'default_device_name'))
            self.opt_target_var_name_var.set(master_config['objective_target'].get('device_variable', 'default_variable_name'))
            
        if 'optimization_steps' in master_config:
            self.opt_steps_var.set(master_config['optimization_steps'])
            
        if 'shots_per_step' in master_config:
            self.shots_per_step_var.set(master_config['shots_per_step'])

        
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
                        self.add_control_with_data(device_name, variable_name, min_val, max_val,"NA")
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
    

    def add_control_with_data(self, device_name, variable_name, min_val, max_val,last_value):
        # This function is similar to add_control but includes data to populate the fields
        next_row = len(self.controls) + 2  # calculate the next row like in add_control

        device_entry = tk.Entry(self)
        device_entry.grid(row=next_row, column=0)
        device_entry.insert(0, device_name)  # Insert the data

        variable_entry = tk.Entry(self)
        variable_entry.grid(row=next_row, column=1)
        variable_entry.insert(0, variable_name)  # Insert the data

        min_entry = tk.Entry(self)
        min_entry.grid(row=next_row, column=2)
        min_entry.insert(0, min_val)  # Insert the data

        max_entry = tk.Entry(self)
        max_entry.grid(row=next_row, column=3)
        max_entry.insert(0, max_val)  # Insert the data
        
        last_value_entry = tk.Entry(self)
        last_value_entry.grid(row=next_row, column=4)
        last_value_entry.insert(0, last_value)  # Insert the data

        # Store these controls in the list so we can retrieve the data later or remove controls if needed
        self.controls.append((device_entry, variable_entry, min_entry, max_entry, last_value_entry))

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





