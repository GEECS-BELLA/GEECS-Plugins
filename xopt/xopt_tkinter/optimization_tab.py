import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading

class OptimizationTab(tk.Frame):
    def __init__(self, master=None, backend=None, geecs_interface=None):
        super().__init__(master)
        self.master = master
        self.backend = backend
        self.geecs_interface = geecs_interface
        self.abort_flag = False
        self.resume_flag = False
        self.disable_sets = False
        self.create_widgets()
        self.best_ctrls_dict = {}
        self.optimization_status=''

    def create_widgets(self):
        self.create_plot_widgets()
        self.create_control_widgets()
        
        self.columnconfigure(0, weight=0, minsize=100)  # Here 200 is the desired width in pixels for the controls column
        self.columnconfigure(1, weight=0)  # Right column (with plots) will resize with the window

    def create_control_widgets(self):
        # Create a frame for the control buttons, positioned at the bottom
        self.control_frame = tk.Frame(self)
        self.control_frame.grid(row=2, column=0, rowspan=2, sticky="ew", padx=10, pady=10)    
        self.control_frame.config(width=200)  # Adjust to your desired width    

        # Create the "Run Optimization" button inside the control frame
        self.run_optimization_button = tk.Button(self.control_frame, text="Run Optimization", command=self.run_optimization)
        self.run_optimization_button.grid(row=2, column=0, padx=2, pady=2)
        
        self.abort_button = tk.Button(self.control_frame, text="Abort Optimization", command=self.abort_optimization)
        self.abort_button.grid(row=2, column=1, padx=2, pady=2)
        
        self.resume_button = tk.Button(self.control_frame, text="Resume Optimization", command=self.resume_optimization)
        self.resume_button.grid(row=3, column=0, padx=2, pady=2)
        
        self.reset_button = tk.Button(self.control_frame, text="Reset Optimization", command=self.reset_optimization)
        self.reset_button.grid(row=3, column=1, padx=2, pady=2)
        
        self.best_value_var = tk.StringVar()
        self.best_value_label = tk.Label(self.control_frame, textvariable=self.best_value_var)
        self.best_value_label.grid(row=2, column=2, padx=2, pady=2)
        self.best_value_var.set("Current Best Value: N/A\nControl Positions: N/A")
        
        self.reset_button = tk.Button(self.control_frame, text="Set to best", command=self.set_to_best)
        self.reset_button.grid(row=3, column=2, padx=2, pady=2)
        
        # Add a StringVar and Label for the optimization step
        self.optimization_step_var = tk.StringVar()
        self.optimization_step_label = tk.Label(self.control_frame, textvariable=self.optimization_step_var)
        self.optimization_step_label.grid(row=4, column=1, padx=2, pady=2)
        self.optimization_step_var.set("Optimization Step: N/A")
        
        # Add a StringVar and Label for the shot number
        self.shot_number_var = tk.StringVar()
        self.shot_number_label = tk.Label(self.control_frame, textvariable=self.shot_number_var)
        self.shot_number_label.grid(row=4, column=2, padx=2, pady=2)
        self.shot_number_var.set("Shot Number: N/A")
        
        # Add a StringVar and Label for the shot number
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, padx=2, pady=2)
        self.status_var.set("Status: N/A")
        
    def create_plot_widgets(self):
        # Create a container frame for the first plot
        self.plot_container1 = tk.Frame(self, bg="white")
        self.plot_container1.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.plot_container1.grid_rowconfigure(0, weight=1)
        self.plot_container1.grid_columnconfigure(0, weight=1)
    
        # Create another container for the second plot
        self.plot_container2 = tk.Frame(self, bg="white")
        self.plot_container2.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.plot_container2.grid_rowconfigure(0, weight=1)
        self.plot_container2.grid_columnconfigure(0, weight=1)

        # Create the first figure with the desired size
        # self.fig1 = plt.Figure()
        self.fig1 = plt.Figure(figsize=(5, 3.1))
        self.fig1.subplots_adjust(left=0.15, right=.95, top=0.9, bottom=0.15)
        
        # self.fig1.tight_layout()
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title("Optimization Progress")
        self.ax1.set_xlabel("Iteration Number")
        self.ax1.set_ylabel("Target Function")

        # Add the canvas to the first plot container
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.plot_container1)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Create the second figure with the desired size
        self.fig2 = plt.Figure(figsize=(5, 3.1))
        self.fig2.subplots_adjust(left=0.15, right=.95, top=0.9, bottom=0.15)
        
        # self.fig2.tight_layout()
        
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("Control Parameters Evolution")
        self.ax2.set_xlabel("Iteration Number")
        self.ax2.set_ylabel("Control Parameter Value")

        # Add the canvas to the second plot container
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.plot_container2)
        self.canvas2.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    
    

    def update_best_value_label(self, best_value, control_positions):
        # Convert the control_positions dictionary to a string format
        control_positions_str = "\n".join(f"{key}: {value}" for key, value in control_positions.items())
    
        # Update the StringVar
        self.best_value_var.set(f"Current Best Value: {best_value}\nControl Positions:\n{control_positions_str}")   
        
    def run_optimization(self):
        # Start the optimization in a new thread
        self.optimization_thread = threading.Thread(target=self.optimization_process)
        self.optimization_thread.start()
    
    def optimization_process(self):
        try:
            config_params = self.backend.config_params
            normalize = config_params.get('normalize', False)
            opt_method = config_params.get('opt_method', '')
        
            print('opt_method:', opt_method)
            print('normalize:', normalize)
        
            yaml_config = self.backend.configure_yaml(self.geecs_interface.backend_vocs)
        
            if self.resume_flag:
                print("continuing existing optimization")
            else:
                self.status_var.set(f"Optimization Status: Intializing")
                self.backend.initialize_xopt()
                self.optimization_step=0
                
            self.status_var.set(f"Optimization Status: Initialized")
        
            for i in range(self.backend.config_params['opt_steps']):  
                if self.abort_flag:
                    print("Optimization aborted by user.")
                    self.abort_flag = False
                    break
                
                print('optimization step: ', i)
                # Update the optimization step in the GUI
                self.optimization_step_var.set(f"Optimization Step: {i + 1}/{self.backend.config_params['opt_steps']}")
                
                result = self.backend.xopt_step()

                if result is None or result.empty:
                    print("No results returned from the optimization step.")
                    continue

                # Update the plots
                self.update_plots(result, yaml_config)
                
                best=dict(result.iloc[result["f"].argmax()]);
                best['f']=round(best['f'],4)
                
                if self.backend.config_params['normalize']: 
                    for key in self.backend.yaml_config['vocs']['variables'].keys():
                        unnormalized_value = self.geecs_interface.unnormalize_controls(key, best[key])
                        self.best_ctrls_dict[key] = round(unnormalized_value, 4)
                
                self.update_best_value_label(best['f'], self.best_ctrls_dict)
                self.status_var.set(f"Optimization Status: Running")

                # Update the GUI
                self.update()
                
            self.status_var.set(f"Optimization Status: Finished")
                

        except Exception as e:
            print(f"An error occurred during the optimization process: {e}")
    
    def update_plots(self, result, yaml_config):
        if 'f' not in result.columns:
            raise ValueError("Result data does not contain 'f' column.")

        # Extract 'f' values
        y1 = result['f'].values

        # Update the first plot
        self.ax1.clear()
        self.ax1.plot(y1)
        self.ax1.set_title("Optimization Progress")
        self.ax1.set_xlabel("Iteration Number")
        self.ax1.set_ylabel("Target Function")

        # Prepare for the second plot
        self.ax2.clear()

        # Safely get control names, if they don't exist, raise an error
        if 'vocs' in yaml_config and 'variables' in yaml_config['vocs']:
            control_names = yaml_config['vocs']['variables'].keys()
        else:
            raise ValueError("'vocs' or 'variables' not found in yaml_config.")

        # Loop through each control and plot it, if it exists in the result DataFrame
        for control in control_names:
            if control in result.columns:
                y2 = result[control].values
                self.ax2.plot(y2, label=control)
            else:
                print(f"Warning: Control '{control}' not found in result data.")

        self.ax2.set_title("Control Parameters Evolution")
        self.ax2.set_xlabel("Iteration Number")
        self.ax2.set_ylabel("Control Parameter Value")
        self.ax2.legend()

        self.canvas1.draw()
        self.canvas2.draw()

    def abort_optimization(self):
        self.abort_flag = True
        print('aborted')
        # Optionally, you can join the thread here to ensure it's finished before doing anything else
        # self.optimization_thread.join()
    
    def set_to_best(self):
        self.geecs_interface.set_from_dict(self.best_ctrls_dict)
        
    def resume_optimization(self):
        self.resume_flag = True
        self.run_optimization()
        print('resuming')
        
    def reset_optimization(self):
        self.resume_flag = False
        print('resetting')

