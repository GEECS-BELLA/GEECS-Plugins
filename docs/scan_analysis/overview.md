# Scan Analysis

Scan Analysis is a comprehensive toolkit for analyzing complete experimental scans, providing capabilities that go beyond single-image analysis to correlate data across multiple devices and shots. It's designed to work seamlessly with any GEECS experiment and often incorporates Image Analysis for processing individual images within scans.

## Overview

While Image Analysis focuses on individual images, Scan Analysis operates at the scan level, providing tools for:

- **Cross-Device Analysis**: Correlate data from multiple experimental devices
- **Statistical Analysis**: Analyze trends and patterns across entire scans
- **Automated Processing**: Discover and analyze scans matching specific criteria
- **Integration**: Combine image analysis with scalar data for comprehensive insights

The framework is built around extensible analyzers that can be customized for specific experimental needs while maintaining compatibility with the broader GEECS ecosystem.

## Key Features

### Multi-Device Correlation
- **Device Synchronization**: Align data from multiple devices using timestamps
- **Cross-Parameter Analysis**: Correlate beam parameters with device settings
- **Statistical Aggregation**: Average, bin, and process data across scan steps
- **Data Fusion**: Combine image data with scalar measurements

### Automated Scan Processing
- **Scan Discovery**: Automatically find scans matching specified criteria
- **Batch Analysis**: Process multiple scans with consistent analysis routines
- **Real-Time Processing**: Analyze scans as they complete
- **Quality Control**: Automated validation and error detection

### Extensible Framework
- **Custom Analyzers**: Easy development of experiment-specific analysis routines
- **Base Classes**: Robust foundation for analyzer development
- **Configuration Management**: Flexible parameter and setting management
- **Plugin Architecture**: Modular design for easy extension

### Integration Capabilities
- **Image Analysis Integration**: Seamlessly incorporate single-image analysis
- **Google Docs Logging**: Automatic upload of results to experiment logs
- **Data Export**: Multiple output formats for further analysis
- **Visualization**: Comprehensive plotting and diagnostic tools

## Architecture

### Core Components

#### Base Classes
- **ScanAnalyzer**: Foundation class for all scan analyzers
- **Configuration Management**: Parameter handling and validation
- **Data Loading**: Unified interface for accessing scan data

#### Analyzer Framework
- **Experiment-Specific Analyzers**: Located in `scan_analysis/analyzers/EXPERIMENT/`
- **Generic Analyzers**: Reusable analysis components
- **Mapping System**: Automatic analyzer selection based on available devices

#### Integration Layer
- **GEECS-Data-Utils Integration**: Seamless access to scan data structures
- **Image Analysis Integration**: Incorporate single-image analysis results
- **Device Management**: Handle multiple device types and data formats

## Creating Custom Analyzers

Scan Analysis makes it easy to create custom analyzers for your experiment:

### Basic Analyzer Structure

```python
from scan_analysis.base import ScanAnalyzer
from image_analysis.utils import read_imaq_png_image
from geecs_data_utils import ScanData

class MyCustomAnalyzer(ScanAnalyzer):
    def __init__(self, scan_tag, device_name=None, skip_plt_show=True, image_analyzer=None):
        super().__init__(scan_tag, device_name=device_name, 
                        skip_plt_show=skip_plt_show, image_analyzer=image_analyzer)

    def run_analysis(self, config_options=None):
        # Access scan data
        scan_data = self.scan_data
        auxiliary_data = self.auxiliary_data  # Pandas DataFrame of sfile data
        
        # Perform analysis
        results = self.analyze_scan_data()
        
        # Save results
        self.append_to_sfile('new_column', results)
        
        # Handle plotting
        self.close_or_show_plot()
        
        return self.display_contents
```

### Analyzer Registration

Register your analyzer by creating a mapping file:

```python
# scan_analysis/mapping/map_MyExperiment.py
from scan_analysis.base import AnalyzerInfo as Info
from .analyzers.MyExperiment.my_analyzer import MyCustomAnalyzer

EXPERIMENT_ANALYZERS = [
    Info(analyzer_class=MyCustomAnalyzer,
         requirements={'UC_MyCamera'}),  # Requires this device
    
    Info(analyzer_class=MyCustomAnalyzer,
         requirements={'OR': ['UC_Camera1', 'UC_Camera2']}),  # Requires either camera
    
    Info(analyzer_class=MyCustomAnalyzer,
         requirements={'AND': ['UC_Camera', {'OR': ['UC_ICT1', 'UC_ICT2']}]}),  # Complex requirements
]
```

## Use Cases

### Beam Parameter Correlation
Analyze relationships between beam parameters and experimental settings:

```python
# Correlate beam size with quadrupole settings
analyzer = BeamSizeCorrelationAnalyzer(scan_tag)
correlation_results = analyzer.run_analysis()
```

### Multi-Shot Statistics
Process statistical variations across scan steps:

```python
# Analyze pointing stability across a parameter scan
analyzer = PointingStabilityAnalyzer(scan_tag)
stability_metrics = analyzer.run_analysis()
```

### Automated Quality Control
Detect and flag problematic scans:

```python
# Automatically detect scans with beam loss or instability
analyzer = QualityControlAnalyzer(scan_tag)
quality_report = analyzer.run_analysis()
```

## Integration with Other Tools

### Image Analysis Integration
Scan Analysis seamlessly incorporates Image Analysis results:

```python
from image_analysis.analyzers import BeamProfileAnalyzer

class ScanBeamAnalysis(ScanAnalyzer):
    def run_analysis(self):
        # Use Image Analysis for individual shots
        image_analyzer = BeamProfileAnalyzer()
        
        for shot in self.scan_shots:
            image_results = image_analyzer.analyze(shot.image)
            # Process results across the scan
```

### Google Docs Integration
Automatically upload results to experiment logs:

```python
# Results automatically uploaded to Google Docs
analyzer.enable_google_docs_upload()
results = analyzer.run_analysis()
```

## Getting Started

1. **Installation**: Follow the [Installation & Setup](installation.md) guide
2. **Analyzer Development**: Review [Creating Analyzers](creating_analyzers.md) for detailed development guide
3. **API Reference**: Use the [API Reference](api.md) for complete documentation

## Related Documentation

- [Installation & Setup](installation.md) - Setup instructions and environment configuration
- [Creating Analyzers](creating_analyzers.md) - Comprehensive guide to analyzer development
- [API Reference](api.md) - Complete API documentation

## Best Practices

### Development Guidelines
- **Single Responsibility**: Each analyzer should focus on one specific analysis task
- **Image Analysis Integration**: Use Image Analysis for single-image processing, Scan Analysis for scan-level insights
- **Error Handling**: Implement robust error handling for device failures and missing data
- **Documentation**: Document analyzer requirements and expected outputs

### Performance Considerations
- **Batch Processing**: Process multiple scans efficiently
- **Memory Management**: Handle large datasets appropriately
- **Caching**: Cache expensive computations when possible

---

*Designed to work with any GEECS experiment, with the flexibility to handle diverse experimental requirements and data structures.*
