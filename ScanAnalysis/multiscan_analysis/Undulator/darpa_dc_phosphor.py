from geecs_python_api.analysis.scans.scan_data import ScanData
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

start_scan = 50
stop_scan = 100

plt.figure(figsize=(10, 10))
for i in np.arange(start=start_scan, stop=stop_scan):
    tag = ScanData.get_scan_tag(year=2025, month=3, day=28, number=i, experiment='Undulator')
    if ScanData.get_scan_folder_path(tag=tag).exists():
        print(tag)
        scan_data = ScanData(tag=tag)
        auxiliary_file_path = scan_data.get_analysis_folder().parent / f"s{tag.number}.txt"
        try:
            auxiliary_data = pd.read_csv(auxiliary_file_path, delimiter='\t')

            total_counts = auxiliary_data[f'UC_DiagnosticsPhosphor Total Counts']
            if np.max(total_counts) > 3.5e6:
                plt.plot(total_counts, label=tag.number)

        except (KeyError, FileNotFoundError) as e:
            logging.warning(f"{e}. Scan parameter not found in auxiliary data. Possible aborted scan. Skipping")

plt.legend()
plt.show()
