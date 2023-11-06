from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class OptimizationTab(QWidget):
    def __init__(self, backend=None, geecs_interface=None):
        super().__init__()

        self.backend = backend
        self.geecs_interface = geecs_interface

        # Main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create a container for the plots
        self.plots_layout = QHBoxLayout()

        # Create the first plot
        self.fig1 = Figure(figsize=(4, 4), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_xlabel("Iteration Number")
        self.ax1.set_ylabel("Target Function")
        self.canvas1 = FigureCanvas(self.fig1)
        self.plots_layout.addWidget(self.canvas1)

        # Create the second plot
        self.fig2 = Figure(figsize=(4, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_xlabel("Iteration Number")
        self.ax2.set_ylabel("Control Parameters")
        self.canvas2 = FigureCanvas(self.fig2)
        self.plots_layout.addWidget(self.canvas2)

        # Add the plots layout to the main layout
        self.main_layout.addLayout(self.plots_layout)

        # Create a frame for the control buttons
        self.control_frame = QFrame()
        self.control_layout = QHBoxLayout(self.control_frame)

        # Create the "Run Optimization" button inside the control frame
        self.run_optimization_button = QPushButton("Run Optimization")
        self.run_optimization_button.clicked.connect(self.run_optimization)
        self.control_layout.addWidget(self.run_optimization_button)

        # Add the control frame to the main layout
        self.main_layout.addWidget(self.control_frame)

    def run_optimization(self):
        try:
            # Safely get configuration parameters with default values if they don't exist
            config_params = self.backend.config_params
            normalize = config_params.get('normalize', False)  # Default to False if not present
            opt_method = config_params.get('opt_method', '')  # Default to empty string if not present
            print('opt_method:', opt_method)
            print('normalize:', normalize)
        
            yaml_config = self.backend.configure_yaml(self.geecs_interface.backend_vocs)
            self.backend.initialize_xopt()
        
            for i in range(30):  # Simulate 30 steps of optimization
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

                # Redraw the canvases and process the GUI events
                self.canvas1.draw_idle()  # Use draw_idle instead of draw to update the canvas in the GUI's idle loop
                self.canvas2.draw_idle()  # Use draw_idle instead of draw to update the canvas in the GUI's idle loop

                QApplication.processEvents()  # Process the GUI events. This line is important to allow the GUI to update the plots

        except Exception as e:
            # Log the exception and provide feedback
            print(f"An error occurred during the optimization process: {e}")
            # Here, you could also log the exception to a logging service, or show a message box to the user, etc.
