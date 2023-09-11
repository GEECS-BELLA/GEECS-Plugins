"""
Mon 8-7-2023

Module for containing functions to both calculate the energy axis calibration and retrieve the previously found fit
parameters.  Works for both the HiResMagSpec and the BCaveMagSpec

@Chris
"""

import numpy as np
# import matplotlib.pyplot as plt


def return_energy_axis(pixel_axis, mag_spec_name):
    if mag_spec_name == 'hires':
        energy_axis = return_hires_energy_axis(pixel_axis)
    elif mag_spec_name == 'acave3':
        energy_axis = return_acave3_energy_axis(pixel_axis)
    else:
        print("Invalid Mag Spec Name for Energy Axis!")
        energy_axis = pixel_axis
    return energy_axis


def return_hires_energy_axis(pixel_axis):
    """
    This is calibrated using the 825 mT settings to get a range of 89.368 to 114.861 MeV across 1287 pixels
    """
    mag_field = '825mT'
    if mag_field == '800mT':
        p0 = 8.66599527e+01
        p1 = 1.78007126e-02
        p2 = 1.10546749e-06
    elif mag_field == '825mT':
        p0 = 8.93681013e+01
        p1 = 1.83568540e-02
        p2 = 1.14012869e-06
    else:
        p0 = 0
        p1 = 0
        p2 = 0
    energy_axis = p0 + p1 * pixel_axis + p2 * np.power(pixel_axis, 2)
    return energy_axis


def return_acave3_energy_axis(pixel_axis):
    """
    Calibrated using the energy axis at 251 mT for the range of 82.167 to 605.241 MeV across 1049 pixels
    """
    mag_field = '251mT'
    if mag_field == '251mT':
        coefs_array = [1.06977145e-34, -5.88860941e-31, 1.40247546e-27, -1.88266185e-24, 1.55532229e-21,
                       -8.08854030e-19, 2.58007035e-16, -4.47996520e-14, 2.00337342e-12, 7.02713877e-10,
                       -7.32308369e-08, 7.28657608e-05, 8.09162396e-02, 8.21672776e+01]  # 0th order last
    else:
        coefs_array = np.zeros(14)
    energy_axis = np.poly1d(coefs_array)(pixel_axis)
    return energy_axis


def read_double_array(file_path):
    try:
        double_array = np.loadtxt(file_path, delimiter='\t')
        return double_array
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return np.array([])
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.array([])


if __name__ == "__main__":
    # There is a lost "triple" somewhere for the three normal-res B-cave mag spec cams
    mode = 'acave3'
    if mode == 'hires':
        super_path = 'Z:/data/Undulator/Y2023/08-Aug/23_0802/auxiliary data/HiResMagCam/'
        filename = 'energy vs pixel HiResMagCam 825 mT.tsv'
        # filename = 'energy vs pixel HiResMagCam 800 mT.tsv'
        energy_file = super_path + filename
        energy_array = read_double_array(energy_file)
        if energy_array.size > 0:
            print("Array of doubles:", energy_array)
            axis = np.arange(0, len(energy_array))
            fit = np.polyfit(axis, energy_array, 2)
            print("Fit: (0th order last)")
            print(" ", fit)

            func = np.zeros(len(energy_array))
            label = ''
            for i in range(len(fit)):
                func = func + fit[i] * np.power(axis, len(fit) - i - 1)
                label = label + "{:.3e}".format(fit[i]) + '*p^' + str(len(fit) - i - 1)
                if i != (len(fit) - 1):
                    label = label + ' + '

    elif mode == 'acave3':
        super_path = 'Z:/data/Undulator/Y2023/09-Sep/23_0905/auxiliary data/'
        filename = 'energy vs pixel UC_ACaveMagSpecCam3 251mT'
        energy_file = super_path + filename
        energy_array = read_double_array(energy_file)
        if energy_array.size > 0:
            print("Array of doubles:", energy_array)
            print("Length: ", len(energy_array))
            axis = np.arange(0, len(energy_array))

            # order_arr = np.arange(10, 16, 1)
            order_arr = [13]
            for order in order_arr:
                fit = np.polyfit(axis, energy_array, order)
                print("Fit: (0th order last)")
                print(" ", fit)
            """
                func = np.zeros(len(energy_array))
                label = ''
                for i in range(len(fit)):
                    func = func + fit[i] * np.power(axis, len(fit) - i - 1)
                    label = label + "{:.3e}".format(fit[i]) + '*p^' + str(len(fit) - i - 1)
                    if i != (len(fit) - 1):
                        label = label + ' + '
            
                plt.plot(axis, energy_array, label="Saved Energy Array")
                plt.plot(axis, np.poly1d(fit)(axis), ls='--', label=str(len(fit) - 1) + "th Polynomial fit")
                plt.legend()
                plt.show()

                plt.semilogy(axis, np.abs(energy_array - np.poly1d(fit)(axis)), label=str(order))
            plt.legend(title="Order")
            plt.ylim([1e-4, 1])
            plt.ylabel("Error Difference (MeV)")
            plt.xlabel("Pixel")
            plt.show()
            """
            print(min(energy_array), max(energy_array))
