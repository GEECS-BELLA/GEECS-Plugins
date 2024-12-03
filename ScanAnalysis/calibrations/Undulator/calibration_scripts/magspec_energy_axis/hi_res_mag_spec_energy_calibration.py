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
    super_path = 'Z:/data/Undulator/Y2023/08-Aug/23_0802/auxiliary data/HiResMagCam/'
    filename = 'energy vs pixel HiResMagCam 825 mT.tsv'
    # filename = 'energy vs pixel HiResMagCam 800 mT.tsv'
    energy_file = super_path + filename
    energy_array = read_double_array(energy_file)
    func = []
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

    plt.plot(energy_array)
    plt.plot(func, ls='--')
    plt.show()
