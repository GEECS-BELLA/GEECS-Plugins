# Creating Custom Analyzers

This comprehensive guide covers everything you need to know about creating custom analyzers for Scan Analysis, from basic concepts to advanced integration patterns.

## Overview

Custom analyzers are the heart of Scan Analysis, allowing you to implement experiment-specific analysis routines that automatically process scans as they complete. Each analyzer is a Python class that inherits from the base `ScanAnalyzer` class and implements specific analysis logic for your experimental needs.

## Quick Start

### Minimal Analyzer Example

Here's the simplest possible analyzer:

```python
from scan_analysis.base import ScanAnalyzer
from geecs_data_utils import ScanData

class MyCustomAnalyzer(ScanAnalyzer):
    def __init__(self, scan_tag, device_name=None, skip_plt_show=True, image_analyzer=None):
        super().__init__(scan_tag, device_name=device_name,
                        skip_plt_show=skip_plt_show, image_analyzer=image_analyzer)

    def run_analysis(self, config_options=None):
        # Your analysis code here
        print(f"Analyzing scan: {self.scan_tag}")

        # Access scan data
        data = self.auxiliary_data  # Pandas DataFrame of sfile data

        # Return any files to display (optional)
        return self.display_contents
```

### File Organization

Create your analyzer in the appropriate experiment directory:

```
ScanAnalysis/
├── scan_analysis/
│   ├── analyzers/
│   │   ├── MyExperiment/           # Your experiment name
│   │   │   ├── __init__.py
│   │   │   └── my_analyzer.py      # Your analyzer
│   │   └── Undulator/              # Example existing experiment
│   └── mapping/
│       ├── map_MyExperiment.py     # Analyzer registration
│       └── map_Undulator.py        # Example mapping
```

## Analyzer Development

### Base Class Features

The `ScanAnalyzer` base class provides several useful attributes and methods:

#### Automatic Initialization
```python
def __init__(self, scan_tag, device_name=None, skip_plt_show=True, image_analyzer=None):
    super().__init__(scan_tag, device_name=device_name,
                    skip_plt_show=skip_plt_show, image_analyzer=image_analyzer)

    # Now available:
    # self.scan_data          - ScanData object
    # self.auxiliary_data     - Pandas DataFrame of sfile data
    # self.noscan            - Boolean: True if NoScan
    # self.scan_parameter    - Name of scanned parameter
    # self.binned_param_values - Array of scan parameter values
```

#### Useful Methods
```python
# Plotting control
self.close_or_show_plot()  # Show plot if testing, close if automated

# Data manipulation
self.append_to_sfile('column_name', data_list)  # Add column to sfile

# File management
self.display_contents.append('path/to/plot.png')  # Add file for display
```

### Data Access Patterns

#### Accessing Scan Information
```python
def run_analysis(self, config_options=None):
    # Basic scan info
    scan_folder = self.scan_data.get_folder()
    analysis_folder = self.scan_data.get_analysis_folder()

    # Scan contents
    contents = self.scan_data.get_folders_and_files()
    devices = contents['devices']
    files = contents['files']

    # Scalar data (sfile)
    if not hasattr(self.scan_data, 'data_frame'):
        self.scan_data.load_scalar_data()

    df = self.scan_data.data_frame  # Pandas DataFrame
    tdms_dict = self.scan_data.data_dict  # Raw TDMS data
```

#### Working with Images
```python
from image_analysis.utils import read_imaq_png_image
import os

def run_analysis(self, config_options=None):
    # Find device folder
    device_folder = os.path.join(self.scan_data.get_folder(), 'UC_MyCamera')

    if os.path.exists(device_folder):
        # Process all images in the device folder
        for filename in os.listdir(device_folder):
            if filename.endswith('.png'):
                image_path = os.path.join(device_folder, filename)
                # IMPORTANT: Use this function for LabVIEW images
                image = read_imaq_png_image(image_path)

                # Analyze image
                results = self.analyze_image(image)
```

### Advanced Patterns

#### Multi-Device Analysis
```python
def run_analysis(self, config_options=None):
    # Check for required devices
    contents = self.scan_data.get_folders_and_files()
    devices = contents['devices']

    camera_data = None
    ict_data = None

    # Find camera device
    for camera_name in ['UC_Camera1', 'UC_Camera2']:
        if camera_name in devices:
            camera_folder = os.path.join(self.scan_data.get_folder(), camera_name)
            camera_data = self.load_camera_data(camera_folder)
            break

    # Find ICT device
    for ict_name in ['UC_ICT1', 'UC_ICT2']:
        if ict_name in self.auxiliary_data.columns:
            ict_data = self.auxiliary_data[ict_name].values
            break

    # Correlate data
    if camera_data is not None and ict_data is not None:
        correlation = self.correlate_camera_ict(camera_data, ict_data)
        self.append_to_sfile('camera_ict_correlation', correlation)
```

#### Statistical Analysis Across Scan
```python
def run_analysis(self, config_options=None):
    # Load image data for each scan step
    beam_sizes = []
    centroids = []

    for step in range(len(self.binned_param_values)):
        # Load images for this step
        step_images = self.load_step_images(step)

        # Analyze each image
        step_sizes = []
        step_centroids = []

        for image in step_images:
            size, centroid = self.analyze_beam_image(image)
            step_sizes.append(size)
            step_centroids.append(centroid)

        # Calculate statistics for this step
        beam_sizes.append(np.mean(step_sizes))
        centroids.append(np.mean(step_centroids, axis=0))

    # Save results
    self.append_to_sfile('beam_size_avg', beam_sizes)
    self.append_to_sfile('centroid_x_avg', [c[0] for c in centroids])
    self.append_to_sfile('centroid_y_avg', [c[1] for c in centroids])
```

#### Integration with Image Analysis
```python
from image_analysis.analyzers import BeamProfileAnalyzer

def run_analysis(self, config_options=None):
    # Create image analyzer
    image_analyzer = BeamProfileAnalyzer()

    # Process device images
    device_folder = os.path.join(self.scan_data.get_folder(), self.device_name)

    results = []
    for image_file in sorted(os.listdir(device_folder)):
        if image_file.endswith('.png'):
            image_path = os.path.join(device_folder, image_file)
            image = read_imaq_png_image(image_path)

            # Use Image Analysis
            analysis_result = image_analyzer.analyze(image)
            results.append(analysis_result)

    # Process results across scan
    beam_sizes = [r['beam_size'] for r in results]
    self.append_to_sfile('beam_size', beam_sizes)
```

## Analyzer Registration

### Creating the Mapping File

Create a mapping file to register your analyzers:

```python
# scan_analysis/mapping/map_MyExperiment.py
from scan_analysis.base import AnalyzerInfo as Info
from ..analyzers.MyExperiment.beam_analyzer import BeamAnalyzer
from ..analyzers.MyExperiment.correlation_analyzer import CorrelationAnalyzer

EXPERIMENT_ANALYZERS = [
    # Simple device requirement
    Info(analyzer_class=BeamAnalyzer,
         requirements={'UC_BeamCam'}),

    # Multiple device options (OR)
    Info(analyzer_class=BeamAnalyzer,
         requirements={'OR': ['UC_BeamCam1', 'UC_BeamCam2', 'UC_BeamCam3']},
         device_name='UC_BeamCam1'),  # Pass device name to analyzer

    # Complex requirements (AND + OR)
    Info(analyzer_class=CorrelationAnalyzer,
         requirements={'AND': ['UC_BeamCam', {'OR': ['UC_ICT1', 'UC_ICT2']}]}),

    # Multiple analyzers for same device
    Info(analyzer_class=BeamAnalyzer,
         requirements={'UC_SpecCam'}),
    Info(analyzer_class=SpectrumAnalyzer,
         requirements={'UC_SpecCam'}),
]
```

### Requirement Patterns

#### Simple Requirements
```python
# Requires exactly this device
requirements={'UC_MyDevice'}

# Requires any one of these devices
requirements={'OR': ['UC_Device1', 'UC_Device2', 'UC_Device3']}
```

#### Complex Requirements
```python
# Requires Device1 AND at least one of Device2/Device3
requirements={'AND': ['UC_Device1', {'OR': ['UC_Device2', 'UC_Device3']}]}

# Requires (Device1 OR Device2) AND (Device3 OR Device4)
requirements={'AND': [
    {'OR': ['UC_Device1', 'UC_Device2']},
    {'OR': ['UC_Device3', 'UC_Device4']}
]}
```

### GUI Integration

Register your experiment mapping in the GUI:

```python
# ScanAnalysis/live_watch/scan_analysis_gui/app/ScAnalyzer.py

# Add import
from scan_analysis.mapping.map_MyExperiment import EXPERIMENT_ANALYZERS as MyExperiment_analyzers

# Add to mapping dictionary
EXPERIMENT_TO_MAPPING = {
    'Undulator': Undulator_analyzers,
    'MyExperiment': MyExperiment_analyzers,  # Add your experiment
    # ... other experiments
}
```

## Testing and Development

### Testing Individual Analyzers

Add a test block at the bottom of your analyzer file:

```python
if __name__ == "__main__":
    from geecs_data_utils import ScanData

    # Test on a specific scan
    tag = ScanData.get_scan_tag(year=2024, month=12, day=25, number=1,
                               experiment='MyExperiment')

    analyzer = MyCustomAnalyzer(scan_tag=tag, skip_plt_show=False)
    results = analyzer.run_analysis()

    print("Analysis complete!")
    print(f"Generated files: {results}")
```

Run your analyzer directly:
```bash
cd ScanAnalysis/scan_analysis/analyzers/MyExperiment/
poetry run python my_analyzer.py
```

### Development Best Practices

#### Error Handling
```python
def run_analysis(self, config_options=None):
    try:
        # Your analysis code
        results = self.perform_analysis()
        return self.display_contents

    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
        return []

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return []
```

#### Logging
```python
import logging

class MyAnalyzer(ScanAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_analysis(self, config_options=None):
        self.logger.info(f"Starting analysis of {self.scan_tag}")
        # ... analysis code
        self.logger.info("Analysis complete")
```

#### Configuration Options
```python
def run_analysis(self, config_options=None):
    # Parse configuration
    config = config_options or {}
    threshold = config.get('threshold', 0.5)
    method = config.get('method', 'gaussian')

    # Use configuration in analysis
    if method == 'gaussian':
        results = self.gaussian_analysis(threshold)
    else:
        results = self.alternative_analysis(threshold)
```

## Advanced Topics

### Google Docs Integration

For automatic upload to experiment logs:

```python
def run_analysis(self, config_options=None):
    # Perform analysis
    results = self.analyze_data()

    # Save plots
    plot_path = os.path.join(self.scan_data.get_analysis_folder(), 'analysis_plot.png')
    self.save_plot(plot_path)

    # Add to display contents for Google Docs upload
    self.display_contents.append(plot_path)

    return self.display_contents
```

### Performance Optimization

#### Caching Expensive Operations
```python
import functools

class MyAnalyzer(ScanAnalyzer):
    @functools.lru_cache(maxsize=128)
    def expensive_calculation(self, parameter):
        # Expensive operation that can be cached
        return complex_calculation(parameter)
```

#### Memory Management
```python
def run_analysis(self, config_options=None):
    # Process images in batches to manage memory
    batch_size = 10
    all_images = self.get_image_list()

    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i+batch_size]
        self.process_image_batch(batch)

        # Clear memory
        del batch
        gc.collect()
```

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Wrong - relative imports don't work in testing
from .other_analyzer import HelperClass

# Correct - use absolute imports
from scan_analysis.analyzers.MyExperiment.other_analyzer import HelperClass
```

#### Image Loading Issues
```python
# Wrong - doesn't handle LabVIEW PNG format
import matplotlib.pyplot as plt
image = plt.imread('image.png')

# Correct - handles LabVIEW bit-shifting
from image_analysis.utils import read_imaq_png_image
image = read_imaq_png_image('image.png')
```

#### Data Access Problems
```python
# Check if data is loaded
if not hasattr(self.scan_data, 'data_frame'):
    self.scan_data.load_scalar_data()

# Check if device exists
contents = self.scan_data.get_folders_and_files()
if 'UC_MyDevice' not in contents['devices']:
    print("Required device not found in scan")
    return []
```

### Debugging Tips

1. **Use Test Mode**: Set `skip_plt_show=False` to see plots during development
2. **Print Debug Info**: Add print statements to understand data flow
3. **Check File Paths**: Verify that expected files and folders exist
4. **Validate Data**: Check data types and shapes before processing
5. **Test Edge Cases**: Test with scans that have missing devices or data

---

*This guide covers the essential patterns for creating robust, maintainable analyzers. For more examples, examine the existing analyzers in the `Undulator` experiment directory.*
