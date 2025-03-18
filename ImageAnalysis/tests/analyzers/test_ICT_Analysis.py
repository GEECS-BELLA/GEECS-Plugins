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

DO_PLOT = False


def copy_of_analysis(data, dt, crit_f, calib):
    value = np.array(data)
    bkg = np.mean(value[0:100])

    sinusoidal_background = ict_analysis.get_sinusoidal_noise(data=data, background_region=(0, 2500))

    if DO_PLOT:
        axis = np.arange(len(value))
        plt.plot(axis, value, c='b')
        plt.plot([axis[0], axis[-1]], [bkg, bkg], ls='--', c='k')
        plt.plot(axis, sinusoidal_background, ls='--', c='r')
        plt.show()

    value = value - sinusoidal_background
    value = np.array(ict_analysis.apply_butterworth_filter(value, order=int(1), crit_f=crit_f))
    ind_roi = ict_analysis.identify_primary_valley(value)
    value = np.array(value[ind_roi])
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
    scan_number = 24

    def get_acave_tdms_file(self, shot_number: int):
        device = 'U_UndulatorExitICT'
        data_folder = f'Z:\\data\\Undulator\\Y2025\\03-Mar\\25_0306\\scans\\Scan{self.scan_number:03d}\\{device}\\'
        shot_prefix = f'Scan{self.scan_number:03d}_{device}'
        return data_folder + shot_prefix + f'_{shot_number:03d}' + '.tdms'

    def get_bcave_tdms_file(self, shot_number: int):
        device = 'U_BCaveICT'
        data_folder = f'Z:\\data\\Undulator\\Y2025\\03-Mar\\25_0306\\scans\\Scan{self.scan_number:03d}\\{device}\\'
        shot_prefix = f'Scan{self.scan_number:03d}_{device}'
        return data_folder + shot_prefix + f'_{shot_number:03d}' + '.tdms'

    def test_analyze_image(self):
        total = 100
        acave_charge = np.zeros(total)
        bcave_charge = np.zeros(total)

        do_bcave_comparison = True

        for i in range(100):
            #i = 13#11
            print(f"Shot {i:03d}")

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
                print("    Computation time:", time.time()-start_time, "s")
                assert time.time()-start_time < 0.1
                assert charge_return == ict_analysis.Undulator_Exit_ICT(data, dt=4e-9, crit_f=0.125)
                print("    Charge:", charge_return, "pC")

                acave_charge[i] = charge_return

            except FileNotFoundError:
                acave_charge[i] = 0

            if do_bcave_comparison:
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
                    print("    Computation time:", time.time() - start_time, "s")
                    assert time.time() - start_time < 0.1
                    assert charge_return == ict_analysis.B_Cave_ICT(data, dt=4e-9, crit_f=0.125)
                    print("    Charge:", charge_return, "pC")

                    bcave_charge[i] = charge_return

                except FileNotFoundError:
                    bcave_charge[i] = 0

        if do_bcave_comparison:
            plt.scatter(bcave_charge, acave_charge, c='b', label='all shots')
            plt.plot([0, 200], [0, 200], c='k', ls='--', label='slope = 1')
            plt.legend()
            plt.xlabel("BCaveICT Charge (pC)")
            plt.ylabel("UndulatorExitICT Charge (pC)")
            plt.show()


if __name__ == '__main__':
    #unittest.main()
    analyzer = TestUC_BeamSpot()
    analyzer.test_analyze_image()


