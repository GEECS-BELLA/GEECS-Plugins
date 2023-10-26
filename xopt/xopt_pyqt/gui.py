import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QLabel, QCheckBox, QComboBox, QFileDialog, QMessageBox
)
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QGridLayout



class MyGUI(QWidget):
   
    def __init__(self, backend=None, geecs_interface=None):
        super().__init__()
        self.backend = backend
        self.geecs_interface = geecs_interface
        self.controls = []
        self.control_row = 1  # Initialize the control row

        # Define the widths for your input fields
        self.device_variable_field_width = 100  # or whatever value is appropriate
        self.min_max_field_width = 50  # half the size of device/variable fields
        self.last_value_field_width = 50  # or whatever value is appropriate

        # Main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # GroupBox for controls
        self.controls_group = QGroupBox("Controls")
        self.controls_layout = QVBoxLayout()
        self.controls_group.setLayout(self.controls_layout)
        self.main_layout.addWidget(self.controls_group)

        # GroupBox for input fields
        self.inputs_group = QGroupBox("Input Fields")
        self.inputs_layout = QGridLayout()  # Change to QGridLayout
        self.inputs_group.setLayout(self.inputs_layout)
        # Adjust the group box's size policy to prevent vertical stretching
        self.inputs_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        

        # Adjust the layout's spacing and margins
        self.inputs_layout.setHorizontalSpacing(1)  # reduce horizontal spacing between widgets
        self.inputs_layout.setVerticalSpacing(10)  # if you also want to reduce vertical spacing
        self.inputs_layout.setContentsMargins(1, 1, 1, 1)  # left, top, right, bottom margins

        # # Adjust the group box's size policy to be minimum and fixed
        # self.inputs_group.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        # Adjust the group box's size policy to be preferred, allowing it to resize more freely
        self.inputs_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        

        # # Set the size policy
        # sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        # self.inputs_group.setSizePolicy(sizePolicy)

        self.main_layout.addWidget(self.inputs_group)

        self.create_widgets()
        self.add_control()  # Initialize with one set of control inputs
        self.on_config_change()  # Initialize with the default settings
    
    def create_widgets(self):
        # Layout for buttons
        self.buttons_layout = QHBoxLayout()

        # Add Control button
        self.add_control_button = QPushButton("Add Control")
        self.add_control_button.clicked.connect(self.add_control)
        self.buttons_layout.addWidget(self.add_control_button)

        # Remove Control button
        self.remove_control_button = QPushButton("Remove Control")
        self.remove_control_button.clicked.connect(self.remove_control)
        self.buttons_layout.addWidget(self.remove_control_button)

        # Initialize Controls button
        self.init_controls_button = QPushButton("Initialize Controls")
        
        # Connect to the method that initializes controls
        self.init_controls_button.clicked.connect(self.perform_action)
        self.buttons_layout.addWidget(self.init_controls_button)

        # Load Config button
        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.load_config)
        self.buttons_layout.addWidget(self.load_config_button)

        # Use Normalization checkbox
        self.normalize_check = QCheckBox("Use Normalization")
        self.normalize_check.stateChanged.connect(self.on_config_change)  # Connect to the config change method
        self.buttons_layout.addWidget(self.normalize_check)

        # Optimization method dropdown
        self.opt_method_combo = QComboBox()
        self.opt_method_combo.addItems(['bayes', 'nelder'])
        self.opt_method_combo.currentIndexChanged.connect(self.on_config_change)  # Connect to the config change method
        self.buttons_layout.addWidget(self.opt_method_combo)

        # Add the buttons layout to the controls layout
        self.controls_layout.addLayout(self.buttons_layout)
        
        labels = ["Device Name", "Variable Name", "Min", "Max", "Last Acquired Value"]
        for i, text in enumerate(labels):
            label = QLabel(text)
            label.setAlignment(Qt.AlignLeft)
            self.inputs_layout.addWidget(label, 0, i)  # Adding labels to the first row of the grid

        # Ensure the columns containing the labels and fields don't stretch beyond their content's width
        self.inputs_layout.setColumnStretch(0, 0)
        self.inputs_layout.setColumnStretch(1, 0)
        self.inputs_layout.setColumnStretch(2, 0)
        self.inputs_layout.setColumnStretch(3, 0)
        self.inputs_layout.setColumnStretch(4, 0)

        # Ensure the rows don't stretch vertically and align at the top
        self.inputs_layout.setRowStretch(self.inputs_layout.rowCount(), 0)  # This ensures no vertical stretch for all rows
        self.inputs_layout.setAlignment(Qt.AlignTop)

 
    def add_control(self):
        # control_widget = QWidget()  # Create a new QWidget which will hold the control layout
        # control_layout = QHBoxLayout(control_widget)  # Assign the control layout to this QWidget
        
        device_entry = QLineEdit()
        variable_entry = QLineEdit()
        min_entry = QLineEdit()
        max_entry = QLineEdit()
        last_value_entry = QLineEdit("NA")
        
        # Set the sizes of QLineEdit widgets
        device_entry.setFixedWidth(self.device_variable_field_width)
        variable_entry.setFixedWidth(self.device_variable_field_width)
        min_entry.setFixedWidth(self.min_max_field_width)
        max_entry.setFixedWidth(self.min_max_field_width)
        last_value_entry.setFixedWidth(self.last_value_field_width)

        entries = [device_entry, variable_entry, min_entry, max_entry, last_value_entry]
        for i, entry in enumerate(entries):
            self.inputs_layout.addWidget(entry, self.control_row, i,Qt.AlignLeft)  # Adding entries to the new row in the grid
            # Set the column stretch to minimum
            self.inputs_layout.setColumnStretch(i, 0)

        self.controls.append((device_entry, variable_entry, min_entry, max_entry, last_value_entry))

        self.control_row += 1  # Increment the row number for the next control


    def remove_control(self):
        if self.controls:
            last_control_set = self.controls.pop()
            for widget in last_control_set:
                widget.deleteLater()  # This will remove the widget from the layout and delete it

    def load_config(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select a Config File", "", "TSV Files (*.tsv);;Text Files (*.txt)")
        if not filepath:
            return

        # Clear all current controls
        while self.controls:
            self.remove_control()  # Reuse the remove_control method to delete controls

        try:
            with open(filepath, newline='') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                next(reader, None)  # Skip headers
                for row in reader:
                    if len(row) == 4:
                        device_name, variable_name, min_val, max_val = row
                        self.add_control_with_data(device_name, variable_name, min_val, max_val)
        except Exception as e:
            QMessageBox.critical(self, "Error loading file", str(e))

    def on_config_change(self):
        new_params = {
            'normalize': self.normalize_check.isChecked(),
            'opt_method': self.opt_method_combo.currentText()
        }
        # Assuming your backend has these methods
        self.backend.set_config_params(new_params)
        self.backend.configure_yaml(self.geecs_interface.backend_vocs)

    def add_control_with_data(self, device_name, variable_name, min_val, max_val):
        # This is similar to add_control, but fills in the data
        self.add_control()  # This will add a new set of controls
        last_control_set = self.controls[-1]
        entries = [device_name, variable_name, min_val, max_val]
        for entry, data in zip(last_control_set[:-1], entries):  # We don't fill the last_value_label here
            entry.setText(data)

    def perform_action(self):
        all_controls_data = []
        for device_entry, variable_entry, min_entry, max_entry, last_value_label in self.controls:
            control_data = {
                "device_name": device_entry.text(),
                "variable_name": variable_entry.text(),
                "min_value": min_entry.text(),
                "max_value": max_entry.text(),
                "last_value": last_value_label.text()
            }
            all_controls_data.append(control_data)

        # Assuming your interface has this method
        initialization_results = self.geecs_interface.initialize_all_controls(all_controls_data)

        for control_data, control_widgets in zip(all_controls_data, self.controls):
            device_name = control_data['device_name']
            variable_name = control_data['variable_name']
            last_value = self.geecs_interface.get_last_acquired_value(device_name, variable_name)
            last_value_label = control_widgets[-1]
            last_value_label.setText(str(last_value))

    def closeEvent(self, event):
        # This method is called when the PyQt window is closed
        self.deleteLater()  # Clean up the GUI
        event.accept()  # Accept the close event

if __name__ == "__main__":
    app = QApplication(sys.argv)
    from backend import MyBackend
    from geecs_functions import GeecsXoptInterface
    
    backend = MyBackend()
    geecs_interface = GeecsXoptInterface()

    gui = MyGUI(backend=backend,geecs_interface=geecs_interface)
    gui.show()
    sys.exit(app.exec_())
