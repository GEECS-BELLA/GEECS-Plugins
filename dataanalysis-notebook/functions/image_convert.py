"""
If you have a problem with image intensities changing, use the function in pngTools.py
Or process the image with addcolumn
"""

import imageio
from skimage import img_as_ubyte, exposure
from PIL import Image, ImageDraw, ImageFont
import glob
import numpy as np
import os
from numpy import unravel_index
import cv2
from pathlib import Path
from functions.pngTools import nBitPNG

from functions.scanparameter import get_scanpara


def downsize_img(img, size):
    """Downsize the image file.
    img: PIL image format( ex) = Image.open(file))
    size: if <1, ratio of width. if 1=>, final basewidth
    return: PIL image format
    """
    if size == 1:
        return img
    else:
        if size <= 1:
            wpercent = size
            basewidth = int(size * float(img.size[0]))
        else:
            basewidth = size
            wpercent = (size / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img_low = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img_low

def get_gif(path, f_format, labview, size=0.2, fps=10., rescale=False, img_bkg=False):
    """Make a GIF from all the files with format f_format in the folder.
    path: directory where the images are
    f_format: format in str. ex) 'png', 'jpg'
    size: either the ratio (0< & <=1) or the final pixel of the width
    rescale:
    1) if 1 ~ 10 (int): gamma correction
    2) if 11 ~ 255(int): linear rescale with max the count
    3) if 'log': log scale.
    32bit image will be systematically convrted to 8bit (except option 2). ( 2^32 -1 -> 2^8 -1 )
    4) if 'auto': max count of the first image becomes 255 (8bit)
    """
    
    # get png file paths
    png_files = sorted(glob.glob(path + '/*.' + f_format))

    # list of txts to draw in the image
    txt_draw_list = get_scanval(path, f_format)

    # downsize the image and make a GIF
    int_maps_gif = []
    img_max0 = 0

    for i in range(len(png_files)):
        
        #If image taken with labview system, significan bits has to be taken into account
        if labview:
            img_sbit = nBitPNG(png_files[i]).astype('int')
            img = Image.fromarray(img_sbit)
        else:
            img = Image.open(png_files[i])
        
        #show image mode
        if i==0:
            mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
            print('Image mode: ', img.mode,'(', mode_to_bpp[img.mode], 'bit)')
       
        # downsize
        img_low = downsize_img(img, size)
        
        #convert image(ndarray) to 8bit image format.Rescale if nessesary.
        if rescale == 'auto' and i==0:
            rescale = float(np.max(np.asarray(img_low)))
            print('Rescale image to make count ', rescale, ' max')
        
        if rescale:
            img_low = rescale_8bit(img_low, rescale = rescale, img_bkg=img_bkg)
        
        #add text to the image
        draw = ImageDraw.Draw(img_low)
        font = ImageFont.truetype('functions/Roboto-Regular.ttf', 20)
        draw.text((0, 0), txt_draw_list[i], fill=255, font=font)
                
        #create a list of images
        int_maps_gif.append(img_low)

    # get a file save name
    scan_str = os.path.basename(os.path.dirname(path))
    device_str = os.path.basename(path)
    f_name = scan_str + '_' + device_str + '.gif'
    # save into gif
    imageio.mimsave(path + '/' + f_name, int_maps_gif, duration=1. / fps)
    print(f_name, ' Saved')

    return None


def get_movie():
    image_folder = 'images'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def img_crop(img, length, cx=None, cy=None):
    """
    Crop the image to square (length^2). If cx and cy are not specified, it will set to
    where the max pixel count is.
    I: plt.imread()
    length: pixel sizes
    cx, cy: center pixel of the image
    """
    if not (cx and cy):
        cy, cx = unravel_index(img.argmax(), img.shape)
    r0 = int(length / 2)
    img2 = img[cy - r0:cy + r0, cx - r0:cx + r0]
    return img2

def rescale_8bit(img, rescale=False, img_bkg=False):
    """
    rescale& background option:subtract background, gamma correct, log scale,
    linear scale to match max as 'rescale'. Then converts to uint8 image format.
    if both rescale and img_bkg are False, this only do the bit conversion.
    
    img: PIL image format
    img_bkg: background image to be subtracted. If True, top-left corner (10%x10% of image) is used as bkg. If None, no background subtraction
    rescale:
        1) if 1 ~ 10 (int): gamma correction
        2) if 11 ~ 255(int): linear rescale with max the count
        3) if 'log': log scale.
        32bit image will be systematically convrted to 8bit (except option 2). ( 2^32 -1 -> 2^8 -1 )
        
    return: Image format uint8.
    """
    img = np.asarray(img)
    
    #background subtraction
    if img_bkg:
        if img_bkg == True:
            # calculate the pixel size equivalent to 10% of the image at the right top
            pix_dx = int(img.shape[0] * 0.1)
            pix_dy = int(img.shape[1] * 0.1)
            img_bkg = img[:pix_dx, img.shape[1] - pix_dy:]
        else:
            pix_dx, pix_dy = img_bkg.shape

        # background subtraction
        corner_ave = np.sum(img_bkg) / pix_dx / pix_dy
        img = img - corner_ave

        # negative values to zero
        img[img < 0] = 0
    
    if rescale and isinstance(rescale, int):
        rescale = float(rescale)
    
    if isinstance(rescale, float) and rescale > 10:
        #rescale so that 'rescale' becomes the max count 225 (uint8)
        #print('rescale to make max count ', rescale)
        img = (img / rescale *255).astype(np.uint8)
        #counts are preserved (255 stays 255 after converting to image format)
        img_uint8 = Image.fromarray(img)
    else:
        if isinstance(rescale, float) and rescale <=10:
            # gamma correction
            #print('Apply gamma correction')
            img = exposure.adjust_gamma(img, rescale)
            #convert to image format           
        elif isinstance(rescale, str) and rescale == 'log':
            # Logarithmic scale
            #print('Apply logarithmic scale')
            img = exposure.adjust_log(img, 1)
        else:
            raise NameError('check the correct input for rescale')
            
        # systematically scaled to uint8 ('L') (ex. 610 in 32bit becomes 80 in 8bit)
        img_uint8 = Image.fromarray(img.astype(np.uint8), mode='L') 
            
    return img_uint8


def get_scanval(path, f_format):
    """
    Get a list of txt to daw in the image
    the txt here is the scan value
    """
    # get png file paths
    png_files = sorted(glob.glob(path + '/*.' + f_format))

    # load scan parameters for caption
    path_date = str(Path(path).parents[2])
    scan = int(path.split(os.sep)[-2][-3:])
    df_scan = get_scanpara(path_date, scan)
    if df_scan is None:
        scanpara = None
    else:
        scanpara = df_scan.columns[0]

    txt_draw_list = []
    for i in range(len(png_files)):
        # text to add
        shotnumber = png_files[i][-7:-4]
        if scanpara:
            scanval = str(round(df_scan.iloc[int(shotnumber) - 1, 0], 3))
            txt_draw = shotnumber + ' ' + scanpara + '=' + scanval
        else:
            txt_draw = shotnumber + ' NoScan'
        txt_draw_list.append(txt_draw)

    return txt_draw_list
