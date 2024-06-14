import unittest

from pathlib import Path

# for this import to work offline, a parent of this folder needs to have a folder
# called "user data". The file dialog for Confirmation.INI can be ignored.
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.controls.devices import GeecsDevice


class HtuExpTestCase(unittest.TestCase):
    def test_htu_exp_init(self):
        self.htu = HtuExp(get_info=False)
        self.htu.close()

    def test_htu_exp_read_database(self):
        with HtuExp(get_info=True) as self.htu:
            self.assertEqual(self.htu.exp_name, "Undulator")
            self.assertEqual(GeecsDevice.exp_info['name'], "Undulator")

    def tearDown(self):
        # reset singleton
        if hasattr(HtuExp, 'instance'):
            del HtuExp.instance
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
