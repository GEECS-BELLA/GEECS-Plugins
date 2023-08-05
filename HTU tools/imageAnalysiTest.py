import numpy as np
import itertools

def test_func(image):
    array_2d = np.array(image)
    # Define the 2D array
    #array_2d = np.array([[10, 20, 30, 40, 50], 
    #                 [60, 70, 80, 90, 100],
    #                 [110, 120, 130, 140, 150],
    #                 [160, 170, 180, 190, 200]])

    # Extract a portion of the 2D array (for example, the subarray from rows 1 and 2 and columns 2 and 3)
    #portion_2d = array_2d[1:3, 2:4]

    # Extract a portion of the 2D array (for example, the subarray from rows 1 and 2 and columns 2 and 3)
    portion_2d = array_2d[700:1000,1500:2100]
    
    # Subtract the constant value
    constant_value=10
    subtracted = portion_2d - constant_value

    # Clip all values below zero to zero
    clipped = np.clip(subtracted, 0, None)

    # Total the values
    total = np.sum(clipped)
    
    return [total,total]