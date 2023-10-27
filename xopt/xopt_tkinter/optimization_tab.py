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
        self.geecs_interface=geecs_interface
        self.abort_flag = False
        self.resume_flag = False
        self.disable_sets = False
        
    
        # Create a frame for the control buttons, which will be positioned at the bottom
        self.control_frame = tk.Frame(self)
        self.control_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10) # use grid

        # Create the "Run Optimization" button inside the control frame
        self.run_optimization_button = tk.Button(self.control_frame, text="Run Optimization", command=self.run_optimization)
        self.run_optimization_button.grid(row=0, column=0, padx=10, pady=10) # changed to grid from pack
        
        self.abort_button = tk.Button(self.control_frame, text="Abort Optimization", command=self.abort_optimization)
        self.abort_button.grid(row=0, column=1, padx=10, pady=10)
        
        self.resume_button = tk.Button(self.control_frame, text="Resume/Continue Optimization", command=self.resume_optimization)
        self.resume_button.grid(row=0, column=2, padx=10, pady=10)
        
        self.reset_button = tk.Button(self.control_frame, text="Reset Optimization", command=self.reset_optimization)
        self.reset_button.grid(row=0, column=3, padx=10, pady=10)
        
        
        # Create a container frame for the first plot
        self.plot_container1 = tk.Frame(self, width=500, height=400, bg="white")  # Set the desired width and height
        self.plot_container1.grid_propagate(False)  # Prevent the container from resizing to fit its contents
        self.plot_container1.grid(row=0, column=0, padx=10, pady=10)  # changed to grid

        # Create another container for the second plot
        self.plot_container2 = tk.Frame(self, width=500, height=400, bg="white")  # Set the desired width and height
        self.plot_container2.grid_propagate(False)  # Prevent the container from resizing to fit its contents
        self.plot_container2.grid(row=0, column=1, padx=10, pady=10)  # changed to grid

        # Create the first figure with the desired size
        self.fig1 = plt.Figure(figsize=(4, 4), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_xlabel("Iteration Number")
        self.ax1.set_ylabel("Target Function")

        # Add the canvas to the first plot container
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.plot_container1)
        self.canvas1.get_tk_widget().place(relwidth=1, relheight=1)  # Make the canvas fill the plot container

        # Create the second figure with the desired size
        self.fig2 = plt.Figure(figsize=(4, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_xlabel("Iteration Number")
        self.ax2.set_ylabel("Control Parameters")

        # Add the canvas to the second plot container
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.plot_container2)
        self.canvas2.get_tk_widget().place(relwidth=1, relheight=1)  # Make the canvas fill the plot container



    def create_widgets(self):
        # Create a matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)  # Create one subplot within the figure

        # Embed the figure into the Tkinter window using a canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a button to start the optimization process
        self.optimize_button = tk.Button(self, text="Optimize", command=self.run_optimization)
        self.optimize_button.pack()
        
        
    def run_optimization(self):
        # Start the optimization in a new thread
        self.optimization_thread = threading.Thread(target=self.optimization_process)
        self.optimization_thread.start()
    

    def optimization_process(self):
        try:
            # Safely get configuration parameters with default values if they don't exist
            config_params = self.backend.config_params
            normalize = config_params.get('normalize', False)  # Default to False if not present
            opt_method = config_params.get('opt_method', '')  # Default to empty string if not present
            print('opt_method:', opt_method)
            print('normalize:', normalize)
        
            yaml_config = self.backend.configure_yaml(self.geecs_interface.backend_vocs)
            
            if self.resume_flag:
                print("continuing existing optimization")
            else:
                self.backend.initialize_xopt()
        
            for i in range(self.backend.config_params['opt_steps']):  
                if self.abort_flag:
                       print("Optimization aborted by user.")
                       self.abort_flag = False  # Reset it for the next time
                       break
                print('optimization step: ', i)
                result = self.backend.xopt_step()

                # Check if result is None or empty, and if so, skip to the next iteration
                if result is None or result.empty:
                    print("No results returned from the optimization step.")
                    continue

                # Check if 'f' column exists in the DataFrame
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

                # Update the GUI
                self.update()

        except Exception as e:
            # Log the exception and provide feedback
            print(f"An error occurred during the optimization process: {e}")
            # Here, you could also log the exception to a logging service, or show a message box to the user, etc.
            
    def abort_optimization(self):
        self.abort_flag = True
        print('aborted')
        # Optionally, you can join the thread here to ensure it's finished before doing anything else
        # self.optimization_thread.join()
        
    def resume_optimization(self):
        self.resume_flag = True
        self.run_optimization()
        print('resuming')
        
    def reset_optimization(self):
        self.resume_flag = False
        print('resetting')

