import unittest
import ctypes
import logging
logging.basicConfig()
LOG = logging.getLogger(__name__)


class TestModuleBasics(unittest.TestCase):
    """Basic module tests"""

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_module_import(self):
        import src.khzwave  # noqa: F401
        print("testing")

    def test_version(self):
        import src.khzwave as khzwave  # noqa: F401
        self.assertEqual(khzwave.__version__, '0.1.0')  # noqa: F401


class TestModuleBuild(unittest.TestCase):

    def test_interface_build(self):
        import src.khzwave.wfs as wfs
        interface = wfs.WfsInterface()
        self.assertNotEqual(len(interface.defines), 0)
        self.assertNotEqual(len(interface.functions), 0)
        self.assertIsInstance(interface.defines["WFS_BUFFER_SIZE"], int)
        self.assertIsInstance(interface.functions["WFS_init"], (list, tuple))
        self.assertIsInstance(interface, ctypes.WinDLL)
        self.assertIsInstance(interface.WFS_init, ctypes._CFuncPtr)

class TestInstrumentDisconnected(unittest.TestCase):
    interface = None

    @classmethod
    def setUpClass(cls):
        import src.khzwave.wfs as wfs
        cls.wfs = wfs
        cls.interface = wfs.WfsInterface()

    def test_instrument_disconnected(self):
        instr = self.wfs.WfsInstrument(self.interface)
        self.assertLessEqual(instr.getInstrumentCount(), 0, "Instrument connected; unable to run disconnect test")

class TestInstrument(unittest.TestCase):

    interface = None

    @classmethod
    def setUpClass(cls):
        import src.khzwave.wfs as wfs
        cls.wfs = wfs
        cls.interface = wfs.WfsInterface()
        # TODO Check if instrument connected here?

    def test_instrument_init(self):
        instr = self.wfs.Wfs20Instrument(self.interface)
        self.assertGreaterEqual(instr.getInstrumentCount(), 1, "Unable to test instrument i")


# def load_tests(loader, standard_tests, pattern):
#     """https://docs.python.org/3.7/library/unittest.html?highlight=unittest#load-tests-protocol"""

#     # top level directory cached on loader instance
#     this_dir = os.path.dirname(__file__)
#     package_tests = loader.discover(start_dir=this_dir, pattern=pattern)
#     standard_tests.addTests(package_tests)
#     return standard_tests
