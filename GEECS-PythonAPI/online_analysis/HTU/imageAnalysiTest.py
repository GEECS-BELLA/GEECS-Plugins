import numpy as np

#below is an example of a basic function that can be used with the GEECS point grey driver
#Note the return structure can not be changed without changing the labview driver. Currently,
#it returns a processed image that is used internally, and a 1D array of scalars. The input
#image is the U16 representation of the raw image. The range however is 0-4095
def example(image,background):
    
    #a background is a required input. Labview passes the background specified by GEECS database
    #Some error handling is taken care of on the labview side. If the background is the wrong shape
    #then an array of zeroes is passed with the correct shape
    array_2d = np.array(image) #note: labview marshalls in the image data array as a numpy array
    background=np.array(background)
    array_2d=array_2d.astype(np.float32)
    background=background.astype(np.float32)
    
    use_background=True
    if use_background:
        working_image_array=array_2d-background
    else:
        working_image_array=array_2d

    #do some kind of analysis
    
    # Extract a portion of the 2D array (for example, the subarray from rows 1 and 2 and columns 2 and 3)
    portion_2d = working_image_array[1:200,1:500]
    
    # Subtract the constant value
    constant_value=60
    subtracted = portion_2d - constant_value

    # Clip all values below zero to zero
    clipped = np.clip(subtracted, 0, None)
    
    #Note, it is required to convert the processed image back to uint16 type
    processed_image=clipped.astype(np.uint16)
        

    # Do some analysis on the processed image
    total = np.sum(processed_image)
    scalar_results=[total,total]
    
    lineout=np.array([[total,total],[total,total],[total,total]]).astype(np.float64)
    #lineout=np.array([[1.1,1.1],[1.1,1.1],[1.1,1.1]])
   

    return (processed_image,scalar_results,lineout)
        

    
