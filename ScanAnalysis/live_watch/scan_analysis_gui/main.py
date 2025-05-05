import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication

# sys.path.insert(0, 'C:\\GEECS\\Developers Version\\source\\GEECS-Plugins\\ScanAnalysis')

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live_watch.scan_analysis_gui.utils.exceptions import exception_hook
from live_watch.scan_analysis_gui.app.ScAnalyzer import ScAnalyzerWindow
from geecs_python_api.analysis.scans.scan_data import ScanData

if __name__ == '__main__':
    """Launches the Scan Analysis GUI"""
    sys.excepthook = exception_hook
    ScanData.reload_paths_config(config_path=Path(r'C:\Users\loasis\.config\geecs_python_api\config.ini')) #Added config path manually otherwise it can't find the config file - 4/3/2025 Eugene
    # ScanData.reload_paths_config() #Added config path manually otherwise it can't find the config file - 4/3/2025 Eugene

    app = QApplication(sys.argv)

    application = ScAnalyzerWindow()
    application.show()
    application.raise_()
    application.activateWindow()

    sys.exit(app.exec_())
