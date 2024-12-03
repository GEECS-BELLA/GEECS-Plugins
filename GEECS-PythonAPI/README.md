# Using ScanData for folder operations

`geecs_python_api\analysis\scans\scan_data.py` contains a class called `ScanData` which contains useful functionality for
retrieving folders and data associated with the GEECS file structure.  Import using `from geecs_python_api.analysis.scans.scan_data import ScanData`

To get a "Scan Tag" that formats the year, month, day, and scan number, use
```python
year = 2024  # Can be YYYY or YY, int or string
month = 11   # Can int or string, number or calendar name/abbreviation
day = 21     # Can be int or string
number = 5   # Can be int or string
scan_tag = ScanData.get_scan_tag(year, month, day, number)
```

To get folders associated with the scan tag, need to also provide experiment name:
```python
scan_folder = ScanData.build_scan_folder_path(scan_tag, experiment=experiment_name)
```

A `ScanData` instance is created either with a folder or a tag+experiment name:
```python
scan_data = ScanData(folder=scan_folder)
scan_data = ScanData(tag=scan_tag, experiment='Undulator')
```

One can also use the following methods to get various folder/ScanData of importance:
```python
# Get the next scan folder location for the current day:    
next_folder = ScanData.get_next_scan_folder(experiment=experiment_name)

# Create a new ScanData at the location of the next scan, creating the folder in the process
scan_data = ScanData.build_next_scan_data(experiment=experiment_name)

# Get the latest ScanData for the current day
latest_scan_data = ScanData.get_latest_scan_data(experiment=experiment_name)

# Get the latest ScanData for the given day
latest_scan_data = ScanData.get_latest_scan_data(experiment=experiment_name, year=2024, month=11, day=21)
```

With a ScanData class, you have access to a variety of useful functions:
```python
# Get the file paths of the data folder and the analysis folder:
data_folder = scan_data.get_folder()
analysis_folder = scan_data.get_analysis_folder()

# Get a list of devices a files within the scan folder:
contents = scan_data.get_folders_and_files()
device_list = contents['devices']
file_list = contents['files']

# Load the tdms file as a dictionary and the s file as a panda dataframe:
scan_data.load_scalar_data()
tdms_dict = scan_data.data_dict
dataframe = scan_data.data_frame

# Get the scan information:
scan_info = scan_data.load_scan_info()
```