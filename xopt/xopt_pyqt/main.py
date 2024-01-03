import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from gui import MyGUI  # Make sure this is the PyQt version of your MyGUI
from optimization_tab import OptimizationTab  # The PyQt version of OptimizationTab
from backend import MyBackend
from geecs_functions import GeecsXoptInterface

class MainApp(QMainWindow):
    def __init__(self, backend, geecs_interface):
        super().__init__()

        # Set the window title
        self.setWindowTitle("My Application")

        # Create the tab widget
        self.tab_widget = QTabWidget()

        # Create instances of your tabs
        self.tab1 = MyGUI(backend=backend, geecs_interface=geecs_interface)
        self.tab2 = OptimizationTab(backend=backend, geecs_interface=geecs_interface)

        # Add the tabs to the tab widget
        self.tab_widget.addTab(self.tab1, "Setup Controls")
        self.tab_widget.addTab(self.tab2, "Optimization Terms")

        # Set the tab widget as the central widget
        self.setCentralWidget(self.tab_widget)

def main():
    backend = MyBackend()
    geecs_interface = GeecsXoptInterface()

    app = QApplication(sys.argv)
    main_app = MainApp(backend=backend, geecs_interface=geecs_interface)
    main_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    def geecs_measurement(input_dict, normalize=None):
        print(input_dict)
        geecs_interface = GeecsXoptInterface()

        for i in list(input_dict.keys()):
            try:
                set_val = float(input_dict[i])
                print("in geecs measurement")
                print(set_val)
                print(i)
                if normalize:
                    print('trying to unnormalize')
                    set_val = geecs_interface.unnormalize_controls(i, set_val)

                print('set ' + str(i) + ' to ' + str(set_val))
                print(geecs_interface.devices[i])

                # Simulate the set command.
                # self.devices[i]["GEECS_Object"].set(self.devices[i]["variable"], set_val)
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
        value = geecs_interface.calcTransmission(setpoint)

        return {'f': value}

    main()
