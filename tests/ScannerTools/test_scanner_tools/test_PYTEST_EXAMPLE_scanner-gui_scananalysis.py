""" Example scanner tools integration test with pytest

pytest is a testing framework that is very popular due to its simplicity. 

"""

import pytest

from pathlib import Path

from geecs_scanner.data_acquisition.scan_manager import ScanManager
from scan_analysis.execute_scan_analysis import analyze_scan

# Sample function, not starting with "test_" that is used in test cases but not 
# run as a test itself. 
def add(x, y):
    return x + y

# Test cases start with "test_"
def test_add():
    assert add(1, 2) == 3

def test_pass_scan_manager_config_to_scan_analysis():
    # Test case to check if scan manager config is passed to scan analysis
    scan_manager = ScanManager()
    analyze_scan(scan_manager.scan_tag, scan_manager.analyzer_list, debug_mode=False)        

    # use standard assert statements to check that the output is as expected
    assert 1 + 1 == 2
    assert Path("/path/to/analysis/output").exists()

if __name__ == "__main__":
    pytest.main()
