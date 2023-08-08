import numpy as np


#below is an example of a basic function that can be used with the GEECS point grey driver
#Note the return structure can not be changed without changing the labview driver. Currently,
#it returns a processed image that is used internally, and a 1D array of scalars. The input
#image is the U16 representation of the raw image. The range however is 0-4095
def example(image):
    array_2d = np.array(image) #note: labview marshalls in the image data array as a numpy array
    array_2d=array_2d.astype(np.float32)

    #do some kind of analysis
    # Extract a portion of the 2D array (for example, the subarray from rows 1 and 2 and columns 2 and 3)
    portion_2d = array_2d[1:200,1:500]
    
    # Subtract the constant value
    constant_value=1
    subtracted = portion_2d - constant_value

    # Clip all values below zero to zero
    clipped = np.clip(subtracted, 0, None)
    
    #Note, it is required to convert the processed image back to uint16 type
    processed_image=clipped.astype(np.uint16)
        

    # Do some analysis on the processed image
    total = np.sum(processed_image)
    scalar_results=[total,total]

    return (processed_image,scalar_results)
        

    
def magCam1(image):

    array_2d = np.array(image)
    array_2d=array_2d.astype(np.float32)

    # Extract a portion of the 2D array (for example, the subarray from rows 1 and 2 and columns 2 and 3)
    portion_2d = array_2d[1:200,1:500]
    
    # Subtract the constant value
    constant_value=1
    subtracted = portion_2d - constant_value

    # Clip all values below zero to zero
    clipped = np.clip(subtracted, 0, None)
    processed_image=clipped.astype(np.uint16)
        

    # Total the values
    total = np.sum(processed_image)
    scalar_results=[total,total]

    return (processed_image,scalar_results)
    
def magCam1Old(image,step):

    if step==0:
        array_2d = np.array(image)
        array_2d=array_2d.astype(np.float32)

        # Extract a portion of the 2D array (for example, the subarray from rows 1 and 2 and columns 2 and 3)
        portion_2d = array_2d[1:200,1:500]
        
        # Subtract the constant value
        constant_value=1
        subtracted = portion_2d - constant_value

        # Clip all values below zero to zero
        clipped = np.clip(subtracted, 0, None)
        result=clipped.astype(np.uint16)
        
    if step==1:
        # Total the values
        total = np.sum(image)
        result=[total,total]
        
def test():
    array_2d=np.array([[1,2,3],[1,2,3],[1,2,3]])
    array_2d=array_2d.astype(np.uint16)
    stat=1.01
    return (array_2d,[stat,stat])