from unittest import TestCase

from pathlib import Path
from geecs_python_api.tools.interfaces.tdms import read_geecs_tdms


class TDMSTestCase(TestCase):
    def test_load_tdms(self):
        test_tdms_path = Path(__file__).parents[2]/"data"/"Scan012.tdms"
        
        # check reading a tdms Path
        data_dict = read_geecs_tdms(test_tdms_path)
        
        # providing a string path should fail on getting Path attributes
        with self.assertRaises(AttributeError):
            data_dict = read_geecs_tdms(str(test_tdms_path))
