"""Test configuration for GEECS-Scanner-GUI.

Patches the GEECS-PythonAPI MySQL database connection at pytest configuration
time (before test collection) so that imports of scan_device.py and other
modules that call GeecsDatabase at module level succeed without a lab network.
"""
