import unittest
import numpy as np
from image_analysis.algorithms.remove_fiducial_cross import FiducialCrossRemover, FiducialCross

class TestFiducialCross(unittest.TestCase):
    def setUp(self):
        self.cross = FiducialCross(x=10, y=10, length=5, thickness=1, angle=45)

    def test_instance(self):
        self.assertIsInstance(self.cross, FiducialCross)

    def test_image(self):
        shape = (20, 20)
        image = self.cross.image(shape)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, shape)

    @unittest.skip
    def test_detect_cross(self):
        shape = (20, 20)
        image = self.cross.image(shape)
        x, y, angle = self.cross.detect_cross(image)
        self.assertEqual(x, 10)
        self.assertEqual(y, 10)
        self.assertEqual(angle, 45)

    @unittest.skip
    def test_detect_cross_no_cross(self):
        shape = (20, 20)
        image = np.zeros(shape)
        x, y, angle = self.cross.detect_cross(image)
        self.assertIsNone(x)
        self.assertIsNone(y)
        self.assertIsNone(angle)

    @unittest.skip
    def test_detect_and_inpaint_crosses(self):
        shape = (20, 20)
        image = self.cross.image(shape)
        image[10, 10] = 0
        image = self.cross.inpaint(image)
        self.assertEqual(image[10, 10], 255)

class TestFitFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.fcr = FiducialCrossRemover()
        return super().setUp()
    
    def test_fit_plateau(self):
        x_mid = 1.0
        width = 2.3
        
        x = np.arange(-2, 5, 0.1)
        y = 1.0 + 0.1 * np.sin(2*np.pi*x/7.0) + (np.abs(x - x_mid) < width/2.0)

        x_mid_fit, width_fit = self.fcr._fit_plateau(x, y, 1.2, 3.0)

        self.assertAlmostEqual(x_mid, x_mid_fit, places=1)
        self.assertAlmostEqual(width, width_fit, places=1)

    @unittest.skip
    def test_fit_plateau_not_regularly_spaced(self):
        x_mid = 1.0
        width = 2.3
        
        x = np.arange(-2, 5, 0.1)
        x = np.cumsum(1 + 0.75 * np.sin(2*np.pi*(x - -2.0) / 2.0))
        assert np.all(np.diff(x) > 0)
        x = (x - x.min()) / (x.max() - x.min()) * 7.0 - 2.0
        y = 1.0 + 0.1 * np.sin(2*np.pi*x/7.0) + (np.abs(x - x_mid) < width/2.0)

        x_mid_fit, width_fit = self.fcr._fit_plateau(x, y, 1.2, 3.0)

        self.assertAlmostEqual(x_mid, x_mid_fit, places=1)
        self.assertAlmostEqual(width, width_fit, places=1)


if __name__ == '__main__':
    unittest.main()