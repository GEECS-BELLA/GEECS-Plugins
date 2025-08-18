import unittest

from wx import App
from livepostprocessing.gui.livepostprocessing_gui import LivePostProcessingGUI

class TestLivePostProcessingGUI(unittest.TestCase):
    def test_init(self):
        app = App()
        frame = LivePostProcessingGUI()

if __name__ == "__main__":
    unittest.main()
