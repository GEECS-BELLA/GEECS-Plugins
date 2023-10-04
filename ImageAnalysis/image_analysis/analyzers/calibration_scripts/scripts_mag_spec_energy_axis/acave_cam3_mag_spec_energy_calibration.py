import numpy as np
import matplotlib.pyplot as plt


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

        print(min(energy_array), max(energy_array))
