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
    img: Image.open(file)
    size: if <1, ratio of width. if 1=>, final basewidth
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

def get_gif(path, f_format, labview, size=0.2, background=None, fps=10., gamma=None, raw=True):
    """Make a GIF from all the files with format f_format in the folder.
    path: directory where the images are
    f_format: format in str. ex) 'png', 'jpg'
    f_name: GIF file name (str)
    size: either the ratio (0< & <=1) or the final pixel of the width
    scale_max: normalize and scale to set the max to scale_max (int)
    background: None or 0 ~1 (if 0.2, background becomes 20% of the saturation)
    gamma: gamma correction if !=1
    raw: if true, no image process, just stitch raw data
    """
    if gamma or background:
        raw = False
    
    # get png file paths
    png_files = sorted(glob.glob(path + '/*.' + f_format))

    # list of txts to draw in the image
    txt_draw_list = get_scanval(path, f_format)

    # downsize the image and make a GIF
    int_maps_gif = []
    img_max0 = 0

    for i in range(len(png_files)):
        #if image taken with labview system, significan bits has to be taken into account
        if labview:
            #img = Image.open(png_files[i])
            img_sbit = nBitPNG(png_files[i]).astype('int')
            #get max count of the first shot for text
            if i==0:
                img_max0 = img_sbit.max()            
            img = Image.fromarray(img_sbit)
        else:
            img = Image.open(png_files[i])
            #get max count of the first shot for text
            if i==0:
                img_max0 = np.asarray(img).max()
        
        # downsize
        img_low = downsize_img(img, size)
        
        if not raw:
            # change from image format to array, unsigned byte format with [0,225]
            im2arr = np.asarray(img_low, np.dtype('uint8'))

            if gamma:
                im2arr = gamma_corr(im2arr, gamma, log=True)
            elif background:
                im2arr = scale_background(im2arr, background)  # rescale the intensity

            im2arr = img_as_ubyte(im2arr)

            # change back to image format
            # gray scale ('L') or rbg (None)
            color_fm = None
            if len(im2arr.shape) == 2:
                color_fm = 'L'
            img_low = Image.fromarray(im2arr, color_fm)
        
        #write text
        draw = ImageDraw.Draw(img_low)
        font = ImageFont.truetype('functions/Roboto-Regular.ttf', 20)
        if raw:
            fill_count = 2000
        else:
            fill_count = 225 #uint8
        draw.text((0, 0), txt_draw_list[i], fill=fill_count, font=font)
                
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


def scale_background(img, scale, img_bkg=None):
    '''
    adjuct the background of the 2d array to scale
    scale: 0 ~1 (if 0.2, background becomes 20% of the saturation)
    img_bkg: background image to be scaled. if None, 10% of the right top of the original image
    '''

    if img_bkg is None:
        # calculate the pixel size equivalent to 10% of the image at the right top
        pix_dx = int(img.shape[0] * 0.1)
        pix_dy = int(img.shape[1] * 0.1)
        img_bkg = img[:pix_dx, img.shape[1] - pix_dy:]
    else:
        pix_dx, pix_dy = img_bkg.shape

    # smooth the backgound
    blur = cv2.GaussianBlur(img_bkg, (11, 11), 0)

    corner_ave = np.sum(blur) / pix_dx / pix_dy

    # the corner is scaled to 'scale'
    scale = 255 * scale / corner_ave

    new_matrix = (img * scale).astype(int)

    return np.array(new_matrix, dtype=img.dtype)


def gamma_corr(img, gamma, log=True, img_bkg=None):
    """
    subtract backgroun, gamma correct, and log scale
    return np array iamge ([0,255])
    img: numpy array with uint8
    """

    if img_bkg is None:
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

    # gamma correction
    img = exposure.adjust_gamma(img, gamma)
    # Logarithmic
    if log:
        img = exposure.adjust_log(img, 1)

    count_max = np.amax(img)
    imgRescale = img / count_max * 255

    new_matrix = (imgRescale).astype(int)

    return new_matrix


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
