"""
Frog Analysis

Quick FROG analysis.

authors:
Kyle Jensen, kjensen@lbl.gov
Finn Kohrell, 
"""
# %% imports

# %% classes

class FrogAnalysis():

    def __init__(self) -> None:
        pass

    def run_analysis(self):
        pass

# %% routine

def testing():

    from geecs_python_api.analysis.scans.scan_date import ScanData

    kwargs = {'year': 2024, 'month': 12, 'day': 10, 'number': 9, 'experiment': 'Undulator'}
    tag = ScanData.get_scan_tag(**kwargs)

    analyzer = FrogAnalysis(scan_tag=tag, device_name="U_FROG_Grenouille-Temporal")

    pass

# %% execute
if __name__ == "__main__":
    testing()


# Finns analysis

# import os
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt


# # folderpath = "12-09-2024 (low power)"
# # subfolder = "U_FROG_Grenouille-Temporal"
# # scan_number = "Scan008"

# folderpath = "12-10-2024 (High Power)"
# subfolder = "U_FROG_Grenouille-Temporal"
# scan_number = "Scan009"


# def load_images(scan_number, base_folder=folderpath):
#     """
#     Load images from the specified scan folder and map them to shot numbers.

#     Parameters:
#         scan_number (str): The scan number to look for.
#         base_folder (str): The base directory containing all scans.

#     Returns:
#         dict: A dictionary mapping shot numbers to (filename, PIL.Image) tuples.
#     """
#     scan_folder = os.path.join(base_folder, scan_number, subfolder)
#     if not os.path.exists(scan_folder):
#         print(f"Error: The folder '{scan_folder}' does not exist.")
#         return {}

#     images = {}
#     for filename in sorted(os.listdir(scan_folder)):
#         if filename.endswith(".png") and scan_number in filename:
#             try:
#                 # Extract shot number from the filename
#                 shot_number = int(filename.split('_')[-1].split('.')[0])
#                 image_path = os.path.join(scan_folder, filename)
#                 img = Image.open(image_path)
#                 images[shot_number] = (filename, img)
#             except Exception as e:
#                 print(f"Error processing file {filename}: {e}")

#     if not images:
#         print(f"No images found in '{scan_folder}' for scan '{scan_number}'.")
#     else:
#         print(f"Loaded {len(images)} images from '{scan_folder}'.")
    
#     return images

# def integrate_images(images):
#     """
#     Integrate each image over the y-axis.

#     Parameters:
#         images (dict): A dictionary of images mapped to shot numbers.

#     Returns:
#         dict: A dictionary of integrated y-axis arrays mapped to shot numbers.
#     """
#     integrated_data = {}
#     for shot_number, (filename, img) in images.items():
#         img_array = np.array(img)
#         y_integrated = np.sum(img_array, axis=1)
#         integrated_data[shot_number] = y_integrated
#     return integrated_data

# def calculate_second_moments(integrated_data):
#     """
#     Calculate the second moment and peak value of each integrated trace.

#     Parameters:
#         integrated_data (dict): A dictionary of integrated y-axis arrays.

#     Returns:
#         dict: A dictionary of second moments and peak values mapped to shot numbers.
#     """
#     second_moments = {}
#     for shot_number, y_integrated in integrated_data.items():
#         y_indices = np.arange(len(y_integrated))
        
#         # Calculate the second moment
#         mean_y = np.sum(y_indices * y_integrated) / np.sum(y_integrated)
#         second_moment = np.sqrt(np.sum(((y_indices - mean_y) ** 2) * y_integrated) / np.sum(y_integrated))
        
#         # Calculate the peak value
#         peak_value = np.max(y_integrated)
        
#         # Store both second moment and peak value
#         second_moments[shot_number] = (second_moment, peak_value)
    
#     return second_moments


# def save_second_moments(second_moments, output_file):
#     """
#     Save the second moments and peak values to a text file.

#     Parameters:
#         second_moments (dict): A dictionary of second moments and peak values mapped to shot numbers.
#         output_file (str): The output file name.
#     """
#     with open(output_file, "w") as f:
#         f.write("Shot_Number\tSecond_Moment\tPeak_Value\n")
#         for shot_number, (second_moment, peak_value) in sorted(second_moments.items()):
#             f.write(f"{shot_number}\t{second_moment}\t{peak_value}\n")
#     print(f"Second moments and peak values saved to '{output_file}'.")

# # Example usage
# images = load_images(scan_number)
# if images:
#     integrated_data = integrate_images(images)
#     second_moments = calculate_second_moments(integrated_data)
#     #save_second_moments(second_moments, output_file=f"second_moments-{folderpath}-{scan_number}.txt")

# def plot_image_with_integration(images, integrated_data, N):
#     if N not in images:
#         print(f"Shot number {N} not found.")
#         return

#     filename, img = images[N]
#     y_integrated = integrated_data[N]
#     img_array = np.array(img)

#     fig, ax1 = plt.subplots(figsize=(5, 4),dpi=300)

#     # Plot the image
#     ax1.imshow(img_array, cmap='gray', aspect='auto')
#     ax1.set_xlabel("X-axis (Pixel)")
#     ax1.set_ylabel("Y-axis (Pixel)")
#     #ax1.set_title(f"Image: {filename}")

#     # Create a twin x-axis for the integrated trace
#     ax2 = ax1.twiny()
#     ax2.plot(y_integrated, range(len(y_integrated)), color='red', label=f"{filename}")
#     ax2.invert_yaxis()  # Match the orientation of the image
#     ax2.set_xlabel("Integrated Value")
#     ax2.legend(loc='upper right')

#     plt.tight_layout()
#     plt.show()


# images = load_images(scan_number)
# if images:
#     integrated_data = integrate_images(images)
#     N = 1  # Set N to the desired shot number
#     plot_image_with_integration(images, integrated_data, N)
