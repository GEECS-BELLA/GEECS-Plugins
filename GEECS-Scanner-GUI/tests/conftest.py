"""Test configuration for GEECS-Scanner-GUI.

Patches the GEECS-PythonAPI MySQL database connection at pytest configuration
time (before test collection) so that imports of scan_device.py and other
modules that call GeecsDatabase at module level succeed without a lab network.
"""

from unittest.mock import MagicMock


def pytest_configure(config):
    """Patch MySQL before any test module is imported."""
    import mysql.connector
    import mysql.connector.pooling

    _mock_conn = MagicMock()
    mysql.connector.connect = MagicMock(return_value=_mock_conn)
    mysql.connector.pooling.MySQLConnectionPool = MagicMock(return_value=_mock_conn)
