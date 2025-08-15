# GEECS Plugin Suite - Docstring Templates

This document provides standardized templates for writing docstrings across all GEECS plugins. We use the NumPy docstring convention for consistency and optimal auto-documentation generation.

## Module-Level Docstring Template

```python
"""
Brief one-line description of the module.

Longer description explaining the module's purpose, main functionality,
and how it fits into the larger GEECS ecosystem.

This module provides functionality for [specific purpose] within the GEECS
plugin suite, including [key features].

Examples
--------
Basic usage example:

>>> from geecs_scanner.app import GEECSScanner
>>> scanner = GEECSScanner()

Notes
-----
Any important notes about dependencies, configuration requirements,
or integration with other GEECS components.

See Also
--------
related_module : Brief description
other_related_module : Brief description
"""
```

## Class Docstring Template

```python
class ExampleClass:
    """
    Brief one-line description of the class.

    Longer description explaining the class purpose, main responsibilities,
    and typical usage patterns within GEECS.

    Parameters
    ----------
    param1 : type
        Description of parameter1
    param2 : type, optional
        Description of parameter2 (default is None)

    Attributes
    ----------
    attribute1 : type
        Description of attribute1
    attribute2 : type
        Description of attribute2

    Methods
    -------
    method1(arg1, arg2)
        Brief description of method1
    method2()
        Brief description of method2

    Examples
    --------
    >>> obj = ExampleClass(param1="value")
    >>> result = obj.method1("arg1", "arg2")

    Notes
    -----
    Any important implementation details, thread safety considerations,
    or integration notes.

    See Also
    --------
    RelatedClass : Brief description
    """
```

## Method/Function Docstring Template

```python
def example_function(param1, param2=None, **kwargs):
    """
    Brief one-line description of the function.

    Longer description explaining what the function does, its role
    in the GEECS workflow, and any important behavior.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, optional
        Description of param2 (default is None)
    **kwargs : dict
        Additional keyword arguments passed to underlying functions

    Returns
    -------
    result_type
        Description of what is returned

    Raises
    ------
    ValueError
        When param1 is invalid
    ConnectionError
        When unable to connect to GEECS devices

    Examples
    --------
    >>> result = example_function("test", param2=42)
    >>> print(result)

    Notes
    -----
    Any performance considerations, side effects, or integration notes.

    See Also
    --------
    related_function : Brief description
    """
```

## Property Docstring Template

```python
@property
def example_property(self):
    """
    Brief description of the property.

    Returns
    -------
    type
        Description of what the property returns

    Notes
    -----
    Any important notes about the property behavior.
    """
```

## Key Guidelines

### 1. First Line
- Always start with a brief, one-line summary
- Use imperative mood ("Calculate the beam centroid" not "Calculates the beam centroid")
- End with a period
- Keep under 79 characters

### 2. Parameters Section
- Use type annotations consistently
- Mark optional parameters clearly
- Provide meaningful descriptions
- Use standard type names (str, int, float, list, dict, etc.)

### 3. Returns Section
- Always document what is returned
- Include the type
- Describe the meaning/content

### 4. Examples Section
- Provide realistic, runnable examples
- Use doctest format when possible
- Show typical usage patterns
- Include expected output when helpful

### 5. Notes Section
- Document side effects
- Mention performance considerations
- Note any GEECS-specific integration details
- Include thread safety information if relevant

### 6. See Also Section
- Reference related functions/classes
- Use consistent formatting
- Help users discover related functionality

## Common Patterns for GEECS

### Device Interface Functions
```python
def connect_device(device_name, timeout=30):
    """
    Establish connection to a GEECS device.

    Parameters
    ----------
    device_name : str
        Name of the device as defined in GEECS database
    timeout : float, optional
        Connection timeout in seconds (default is 30)

    Returns
    -------
    bool
        True if connection successful, False otherwise

    Raises
    ------
    ConnectionError
        When device is unreachable
    ValueError
        When device_name is not in GEECS database
    """
```

### Analysis Functions
```python
def analyze_beam_profile(image, roi=None):
    """
    Analyze beam profile from camera image.

    Parameters
    ----------
    image : numpy.ndarray
        2D array representing the camera image
    roi : tuple of int, optional
        Region of interest as (top, bottom, left, right)

    Returns
    -------
    dict
        Analysis results containing:
        - 'centroid_x' : float, beam centroid x-coordinate
        - 'centroid_y' : float, beam centroid y-coordinate
        - 'beam_size' : float, RMS beam size in pixels
    """
```

### Configuration Classes
```python
class ScanConfig:
    """
    Configuration container for GEECS scan parameters.

    This class encapsulates all parameters needed to configure
    a scan in the GEECS system, including device settings,
    scan ranges, and timing parameters.

    Parameters
    ----------
    device_var : str
        GEECS device variable to scan
    start : float
        Starting value for the scan
    end : float
        Ending value for the scan
    step : float
        Step size for the scan

    Attributes
    ----------
    scan_points : list
        Calculated scan points based on start, end, step
    estimated_duration : float
        Estimated scan duration in seconds
    """
