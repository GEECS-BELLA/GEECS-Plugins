"""
Unit Test made to fix the sinusoidal noise issue in the UndulatorExitICT analysis.  Contains copies of actual functions
so that plotting is available.  (Known issue is that Labview crashes if matplotlib is imported)

TODO currently hardcoded to be Scan 24 of March 6, 2025.  Make more generalized

-Chris
"""

import online_analysis.HTU.picoscope_ICT_analysis as ict_analysis

import unittest
import time

from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DO_PLOT = False
TEST_TIME = False
MAX_TIME = 0.2
FIND_SHOT = 23
SKIP_ACAVE = False
SKIP_BCAVE = False


def copy_of_analysis(data, dt, crit_f, calib):
    value = np.array(data)
    value = np.array(ict_analysis.apply_butterworth_filter(value, order=int(1), crit_f=crit_f))
    bkg = np.mean(value[0:100])

    signal_location = np.argmin(data)
    first_interval_end = signal_location - 100 if signal_location > 100 else None
    second_interval_start = signal_location + 600 if signal_location + 600 < len(value) else None

    sinusoidal_background_1 = ict_analysis.get_sinusoidal_noise(data=value,
                                                              signal_region=(first_interval_end, second_interval_start))
    value_2 = value - sinusoidal_background_1

    sinusoidal_background_2 = ict_analysis.get_sinusoidal_noise(data=value_2,
                                                              signal_region=(first_interval_end, second_interval_start))
    subtracted_value = value_2 - sinusoidal_background_2

    if DO_PLOT:
        axis = np.arange(len(value))
        plt.plot(axis, value, c='b')
        plt.plot([axis[0], axis[-1]], [bkg, bkg], ls='--', c='k')
        plt.plot(axis, subtracted_value, c='orange')
        if second_interval_start:
            plt.plot([second_interval_start, second_interval_start], [np.min(value), np.max(value)], ls='dotted', c='g')
        if first_interval_end:
            plt.plot([first_interval_end, first_interval_end], [np.min(value), np.max(value)], ls='dotted', c='g')
        plt.plot(axis, sinusoidal_background_1 + sinusoidal_background_2, ls='--', c='r')
        plt.show()

    ind_roi = ict_analysis.identify_primary_valley(subtracted_value)
    value = np.array(subtracted_value[ind_roi])
    integrated_signal = np.trapz(value, x=None, dx=dt)
    charge_pC = integrated_signal * -calib * 10 ** (12)

    return charge_pC


def copy_of_B_Cave_ICT(data, dt, crit_f):
    calib = 0.2
    charge_pC = copy_of_analysis(data, dt, crit_f, calib)
    return charge_pC


def copy_of_Undulator_Exit_ICT(data, dt, crit_f):
    calib = 0.2 / 2.78
    charge_pC = copy_of_analysis(data, dt, crit_f, calib)
    return charge_pC

class TestUC_BeamSpot(unittest.TestCase):
    scan_day = 24  # 28  # 7  # 6
    scan_number = 1  # 6  # 56  # 24

    def get_acave_tdms_file(self, shot_number: int):
        device = 'U_UndulatorExitICT'
        data_folder = f'Z:\\data\\Undulator\\Y2025\\03-Mar\\25_03{self.scan_day:02d}\\scans\\Scan{self.scan_number:03d}\\{device}\\'
        shot_prefix = f'Scan{self.scan_number:03d}_{device}'
        return data_folder + shot_prefix + f'_{shot_number:03d}' + '.tdms'

    def get_bcave_tdms_file(self, shot_number: int):
        device = 'U_BCaveICT'
        data_folder = f'Z:\\data\\Undulator\\Y2025\\03-Mar\\25_03{self.scan_day:02d}\\scans\\Scan{self.scan_number:03d}\\{device}\\'
        shot_prefix = f'Scan{self.scan_number:03d}_{device}'
        return data_folder + shot_prefix + f'_{shot_number:03d}' + '.tdms'

    def test_analyze_image(self):
        folder = Path(self.get_bcave_tdms_file(shot_number=0)).parent
        total = len(list(folder.glob('*.tdms')))
        acave_charge = np.zeros(total)
        bcave_charge = np.zeros(total)

        do_bcave_comparison = True
        global DO_PLOT

        for i in range(total):
            if FIND_SHOT is not None:
                DO_PLOT = bool(i == FIND_SHOT-1)
                if not DO_PLOT:
                    continue
            print(f"Shot {i:03d}")

            if not SKIP_ACAVE:
                try:
                    print(f"  A Cave ICT:")
                    filename = self.get_acave_tdms_file(shot_number=i)
                    tdms_file = TdmsFile.read(filename)

                    if print_tdms_stats := False:
                        print(tdms_file.properties)
                        for group in tdms_file.groups():
                            print(group.name, ":")
                            for channel in group.channels():
                                print("-", channel.name)

                    data = tdms_file['Picoscope']['ChB'][:]

                    start_time = time.time()
                    charge_return = copy_of_Undulator_Exit_ICT(data, dt=4e-9, crit_f=0.125)
                    if DO_PLOT is False and TEST_TIME is True:
                        print("    Computation time:", time.time() - start_time, "s")
                        assert time.time()-start_time < MAX_TIME
                    assert charge_return == ict_analysis.Undulator_Exit_ICT(data, dt=4e-9, crit_f=0.125)
                    print("    Charge:", charge_return, "pC")

                    acave_charge[i] = charge_return

                except FileNotFoundError:
                    acave_charge[i] = 0

            if not SKIP_BCAVE:
                try:
                    print(f"  B Cave ICT:")
                    filename = self.get_bcave_tdms_file(shot_number=i)
                    tdms_file = TdmsFile.read(filename)

                    if print_tdms_stats := False:
                        print(tdms_file.properties)
                        for group in tdms_file.groups():
                            print(group.name, ":")
                            for channel in group.channels():
                                print("-", channel.name)

                    data = tdms_file['Picoscope']['ChA'][:]

                    start_time = time.time()
                    charge_return = copy_of_B_Cave_ICT(data, dt=4e-9, crit_f=0.125)
                    if DO_PLOT is False and TEST_TIME is True:
                        print("    Computation time:", time.time() - start_time, "s")
                        assert time.time() - start_time < MAX_TIME
                    assert charge_return == ict_analysis.B_Cave_ICT(data, dt=4e-9, crit_f=0.125)
                    print("    Charge:", charge_return, "pC")

                    bcave_charge[i] = charge_return

                except FileNotFoundError:
                    bcave_charge[i] = 0

            if FIND_SHOT is not None and DO_PLOT:
                return  # Exit here, we are done

        if do_bcave_comparison and not SKIP_ACAVE and not SKIP_BCAVE:
            plt.scatter(bcave_charge, acave_charge, c='b', label='all shots')
            max_range = max(np.max(bcave_charge), np.max(acave_charge))*1.2
            if self.scan_day != 24:
                plt.plot([0, max_range], [0, max_range], c='k', ls='--', label='slope = 1')
            for i in range(len(bcave_charge)):
                plt.text(x=bcave_charge[i], y=acave_charge[i], s=f"{i+1}")
            plt.legend()
            plt.xlabel("BCaveICT Charge (pC)")
            plt.ylabel("UndulatorExitICT Charge (pC)")
            plt.show()

        elif not SKIP_BCAVE:
            shot_arr = np.arange(len(bcave_charge)) + 1
            plt.scatter(shot_arr, bcave_charge, c='b', label='all shots')
            plt.legend()
            plt.xlabel("BCaveICT Charge (pC)")
            plt.ylabel("UndulatorExitICT Charge (pC)")
            plt.show()


if __name__ == '__main__':
    #unittest.main()
    analyzer = TestUC_BeamSpot()
    analyzer.test_analyze_image()


