import numpy as np
from array import array
import HiResAnalysisModules.HiResMagSpecAnalysis as MagSpecAnalysis
import cv2

import png

def load_and_correct_sbit(image_path):
    def get_sbit_from_png(png_file_path):
        """Get the sBit value from a PNG file's header."""
        with open(png_file_path, 'rb') as f:
            reader = png.Reader(file=f)
            for chunk in reader.chunks():
                if chunk[0] == b'sBIT':
                    sbit = chunk[1][0]
                    return sbit
        return None

    def correct_sbit(img, sbit_value):
        """Correct the image using the sBit value."""
        shift_amount = 16 - sbit_value
        corrected_img = img >> shift_amount
        return corrected_img

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Extract the sBit value
    sbit = get_sbit_from_png(image_path)

    # Apply the sBit correction
    if sbit is not None:
        corrected_img = correct_sbit(img, sbit)
    else:
        print("Couldn't find sBit value. Using the original image.")
        corrected_img = img

    return corrected_img
    
def calculate_statistics(image_data):
    """
    Calculate statistical metrics on the provided image data:
    - First moment (mean)
    - Second moment (variance)
    - Full Width at Half Maximum (FWHM)
    - Maximum value
    - Total counts (sum of all pixels)
    """
    
    # Compute the sum of the image data in each direction
    sum_x = np.sum(image_data, axis=0)
    sum_y = np.sum(image_data, axis=1)

    # Calculate the first and second moments
    indices_x = np.arange(sum_x.size)
    indices_y = np.arange(sum_y.size)
    
    mean_x = np.sum(indices_x * sum_x) / np.sum(sum_x) if np.sum(sum_x) != 0 else 0
    mean_y = np.sum(indices_y * sum_y) / np.sum(sum_y) if np.sum(sum_y) != 0 else 0
    
    var_x = np.sqrt(np.sum(sum_x * (indices_x - mean_x)**2) / np.sum(sum_x)) if np.sum(sum_x) != 0 else 0
    var_y = np.sqrt(np.sum(sum_y * (indices_y - mean_y)**2) / np.sum(sum_y)) if np.sum(sum_y) != 0 else 0
    
    # Compute the FWHM
    half_max_x = np.max(sum_x) / 2.0
    half_max_y = np.max(sum_y) / 2.0
    
    indices_half_max_x = np.where(sum_x > half_max_x)[0]
    indices_half_max_y = np.where(sum_y > half_max_y)[0]
    
    fwhm_x = indices_half_max_x[-1] - indices_half_max_x[0] + 1 if indices_half_max_x.size else 0
    fwhm_y = indices_half_max_y[-1] - indices_half_max_y[0] + 1 if indices_half_max_y.size else 0

    # Calculate the maximum value and total counts
    max_value = np.max(image_data)
    total_counts = np.sum(image_data)

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "var_x": var_x,
        "var_y": var_y,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "max_value": max_value,
        "total_counts": total_counts
    }
    
def generate_2d_gaussian(shape, sigma=20):
    """
    Generate a 2D Gaussian distribution.
    
    Parameters:
    - shape: tuple of (height, width) for the desired shape of the Gaussian.
    - sigma: standard deviation of the Gaussian.
    
    Returns:
    - 2D numpy array representing the Gaussian.
    """
    x = np.linspace(-shape[1]//2, shape[1]//2, shape[1])
    y = np.linspace(-shape[0]//2, shape[0]//2, shape[0])
    x, y = np.meshgrid(x, y)
    
    gaussian = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return gaussian

class BaseAnalysis:
    """
    The base class for all image analysis.

    Parameters:
    -----------
    image : numpy.ndarray
        The image to be analyzed.
    background : numpy.ndarray
        The background to be subtracted or used in analysis.
    """

    def __init__(self, image, background):
        """
        Initializes the base analysis class.

        Parameters:
        -----------
        image : ndarray
            The main image data.
        background : ndarray
            The background data for the image.
        """
        array_2d = np.array(image)
        
        # if self.simulate_image:
            # image_shape=array_2d.shape
            # # Generate a 2D Gaussian for the given shape
            # gaussian_data = generate_2d_gaussian(image_shape, sigma=20)
            # # Normalize the Gaussian to have a peak of 4000
            # normalized_gaussian = gaussian_data * 4000 / np.max(gaussian_data)
            # # Convert to uint16
            # array_2d = normalized_gaussian.astype(np.int16)+60
      
            # # Generate a random integer between 1 and 220
            # random_int = np.random.randint(1, 221)
            # # Convert it into a string of length 3 with zeros padded on the left
            # formatted_string = f"{random_int:03}"
            # original_string="Z:\\data\\Undulator\\Y2023\\08-Aug\\23_0809\\scans\\Scan024\\UC_ALineEBeam3\\Scan024_UC_ALineEBeam3_084.png"
            # path_to_image = original_string.replace("084", formatted_string)
            # # Load the image
            # array_2d = load_and_correct_sbit(path_to_image)
            
        self.background = np.array(background).astype(np.int16)
        self.array_2d = array_2d.astype(np.int16)
        
        if self.use_background:
            self.working_image_array = self.array_2d - self.background
        else:
            self.working_image_array = self.array_2d

    @property
    def use_background(self):
        """
        Determines whether to use the background in the analysis.

        Returns:
        --------
        bool
            True if the background should be used, otherwise False.
        """
        return True
        
    @property
    def simulate_image(self):
        """
        If true, the workflow will use a simulated gaussian distribution

        Returns:
        --------
        bool
            True simulation should be used, otherwise False.
        """
        return False
        

    def execute(self):
        """
        Executes the analysis on the given image and background. 
        This method should be overridden in child classes.

        Returns:
        --------
        tuple
            Default data as a tuple containing:
            - 2D uint16 array
            - 1D double array
            - 2D double array
        """
        return (
            np.zeros((2, 2), dtype=np.uint16),
            np.zeros(2, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64)
        )

    def validate_output(self, result):
        """
        Validates and/or provides default values for the output data.

        Parameters:
        -----------
        result : tuple
            The result data to be validated or filled with defaults.

        Returns:
        --------
        tuple
            Validated and possibly filled result data.
        """

        # Default outputs
        default_2d_uint16 = np.zeros((2, 2), dtype=np.uint16)
        default_1d_float64 = np.zeros(2, dtype=np.float64)
        default_2d_float64 = np.zeros((2, 2), dtype=np.float64)

        if not (isinstance(result[0], np.ndarray) and result[0].dtype == np.uint16):
            result = (default_2d_uint16, result[1], result[2])

        if not (isinstance(result[1], np.ndarray) and len(result[1].shape) == 1 and result[1].dtype == np.float64):
            result = (result[0], default_1d_float64, result[2])

        if not (isinstance(result[2], np.ndarray) and len(result[2].shape) == 2 and result[2].dtype == np.float64):
            result = (result[0], result[1], default_2d_float64)

        return result

class UC_ExampleCamAnalysis(BaseAnalysis):
    """
    Analysis class specific for DeviceA.

    Extends:
    --------
    BaseAnalysis
    """
    @property
    def use_background(self):
        return True
            
    def execute(self):
        """
        Executes the device-specific analysis on the given image and background.

        Returns:
        --------
        tuple
            Analysis result data as a tuple containing:
            - 2D uint16 array
            - 1D double array
            - 2D double array
        """
        # ... your device-specific code ...

        result = (
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
        )

        # Ensure the output matches the expected format
        return self.validate_output(result)
  
class HiResMagSpecAnalysisClass(BaseAnalysis):
    """
    Analysis class specific for HiResMagSpec.

    Extends:
    --------
    BaseAnalysis
    """

    @property
    def use_background(self):
        return False

    def execute(self):
        """
        Executes the HiResMagSpec specific analysis on the given image.

        Returns:
        --------
        tuple
            Analysis result data as a tuple containing:
            - 2D uint16 array
            - 1D double array
            - 2D double array
        """
        
        # Factor to go from camera counts to pC/MeV
        # Record: July 25th, Scan 24, HiResMagSpec
        normalization_triggerdelay = 15.497208
        normalization_exposure = 0.010000
        normalization_factor = 1.1301095153900242e-06

        inputParams = {
            "Threshold-Value": 230,
            "Saturation-Value": 4095,
            "Normalization-Factor": normalization_factor,
            "Transverse-Calibration": 43,
            "Transverse-Slice-Threshold": 0.02,
            "Transverse-Slice-Binsize": 5
        }

        # You've used the variable name 'image' in the function, 
        # but in the class, you should be using 'self.working_image_array'
        MagSpecDict = MagSpecAnalysis.AnalyzeImage(self.working_image_array, inputParams)
        values = np.array(list(MagSpecDict.values()), dtype=np.float64)

        result = (
            self.working_image_array.astype(np.uint16),  # Ensure the image is uint16 type
            values,
            None  # Here, you can replace None with your 2D double array, if available
        )

        # Ensure the output matches the expected format and fill defaults if necessary
        return self.validate_output(result)

class Aline3Analysis(BaseAnalysis):
    def __init__(self, image, background):
        """
        Initializes the base analysis class.

        Parameters:
        -----------
        image : ndarray
            The main image data.
        background : ndarray
            The background data for the image.
        """
        if self.simulate_image:
            image_shape=image.shape
            # Generate a 2D Gaussian for the given shape
            gaussian_data = generate_2d_gaussian(image_shape, sigma=20)
            # Normalize the Gaussian to have a peak of 4000
            normalized_gaussian = gaussian_data * 4000 / np.max(gaussian_data)
            # Convert to uint16
            simulated_image = normalized_gaussian.astype(np.int16)+60
        
            # # Generate a random integer between 1 and 220
            # random_int = np.random.randint(1, 221)
            # # Convert it into a string of length 3 with zeros padded on the left
            # formatted_string = f"{random_int:03}"
            # original_string="Z:\\data\\Undulator\\Y2023\\08-Aug\\23_0809\\scans\\Scan024\\UC_ALineEBeam3\\Scan024_UC_ALineEBeam3_084.png"
            # path_to_image = original_string.replace("084", formatted_string)
            # # Load the image
            # simulated_image = load_and_correct_sbit(path_to_image)

            self.working_image_array = simulated_image.astype(np.int16)

    @property
    def use_background(self):
        return False  # For now, we're not using background in Aline3 analysis
    @property
    def simulate_image(self):
        return True

    def execute(self):
        
        # 2. Apply a median filter to the provided image data
        median_filtered_img = cv2.medianBlur(self.working_image_array, 5)
        
        #2.1
        # Find the average of the smallest 50 values
        smallest_50_avg = np.mean(np.partition(median_filtered_img, 50)[:50])

        # Subtract this average from the array
        median_filtered_img = median_filtered_img - smallest_50_avg

        # 3. Binarize using a threshold of "threshold" counts
        max_value = np.iinfo(self.working_image_array.dtype).max
        threshold=50
        _, binarized_img = cv2.threshold(median_filtered_img, threshold, max_value, cv2.THRESH_BINARY)

        # 4. Find the contours
        contours, _ = cv2.findContours(binarized_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the center of the image
        image_center = (binarized_img.shape[1] // 2, binarized_img.shape[0] // 2)

        # Threshold distance for considering a contour "near" the center (adjust as needed)
        threshold_distance = 250

        # Initialize the variable to store the desired contour
        desired_contour = None

        # Iterate through the contours
        for contour in contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
            
            # Calculate the Euclidean distance from the centroid to the image center
            distance_to_center = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
            
            # Check if the contour's centroid is near the image center and if the area is larger than 100
            if distance_to_center < threshold_distance and cv2.contourArea(contour) > 200:
                desired_contour = contour
                break

        # 6. Create a mask using the desired contour (if available)
        contour_mask = np.zeros_like(self.working_image_array)
        if desired_contour is not None:
            cv2.drawContours(contour_mask, [desired_contour], -1, max_value, thickness=cv2.FILLED)


        # 7. Multiply the mask with the provided image data
        masked_image = cv2.bitwise_and(self.working_image_array, contour_mask)

        # 8. Subtract background (if central contour is available)
        if desired_contour is not None:
            boundary_mask = cv2.dilate(contour_mask, np.ones((3,3), np.uint8), iterations=1) - contour_mask
            boundary_avg_corrected_img = np.mean(self.working_image_array[boundary_mask == max_value])
            background_subtracted = masked_image - boundary_avg_corrected_img
            background_subtracted_clipped = np.clip(background_subtracted, 0, max_value)
        else:
            background_subtracted_clipped = masked_image
            
        stats = calculate_statistics(background_subtracted_clipped)
        values = np.array(list(stats.values()), dtype=np.float64)
        
        result = (
            background_subtracted_clipped.astype(np.uint16),  # Ensure the image is uint16 type
            values,
            None  # Here, you can replace None with your 2D double array, if available
        )

        # Ensure the output matches the expected format and fill defaults if necessary
        return self.validate_output(result)

def analyze_image(device_type, image, background):
    """
    Main function to analyze an image based on the device type.

    Parameters:
    -----------
    device_type : str
        Type of the device, e.g., "DeviceA". This is used to 
        select out the specific device from the analyzer
        dictionary
    image : numpy.ndarray
        The image to be analyzed.
    background : numpy.ndarray
        The background to be subtracted or used in analysis.

    Returns:
    --------
    tuple
        Analysis result data as a tuple containing:
        - 2D uint16 array
        - 1D double array
        - 2D double array

    Raises:
    -------
    ValueError
        If the provided device type is not supported.
    """
    analyzers = {
        #"UC_TestCam": HiResMagSpecAnalysisClass,
        "UC_TestCam": Aline3Analysis,
        # ... more devices here
    }

    analyzer_class = analyzers.get(device_type)
    if not analyzer_class:
        raise ValueError(f"Unknown device type: {device_type}")

    analyzer = analyzer_class(image, background)
    return analyzer.execute()

