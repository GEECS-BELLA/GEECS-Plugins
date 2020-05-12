# Show the lastest image of a device saved in the scan data, and update it every second
#This is useful when you want to see images online during a scan
#See main() function for example usage

# Fumika Jan 20, 2020

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import filecmp


def u16_to_8bit(image, display_min=0, display_max=255, autoscale=True):
    # Here I set copy=True in order to ensure the original image is not
    # modified. If you don't mind modifying the original image, you can
    # set copy=False or skip this step.
    if autoscale:
        display_min = image.min()
        display_max = image.max()

    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    np.floor_divide(image, (display_max - display_min + 1) / 256,
                    out=image, casting='unsafe')
    return image.astype(np.uint8)

def listdir_fullpath(d):
    # get a list of directories in fullpath in the folder
    return [os.path.join(d, f) for f in os.listdir(d)]

def imread(path):
    # Read a file
    if path.endswith('.tif'):
        I = plt.imread(path)
        I = u16_to_8bit(I)
    else:
        I = cv2.imread(path, 0)
    return I

def last_img_path(dir_date, device_name):
    # get last image saved
    d_scans = dir_date + '\\scans'
    list_d_scans = listdir_fullpath(d_scans)

    # look for a latest scan forlder which has the device folder
    n_scans = len(list_d_scans)
    d_last_device = ''
    for i in range(n_scans):
        d_device = list_d_scans[n_scans - i - 1] + '\\' + device_name
        if os.path.isdir(d_device):
            d_last_device = d_device
            break
    # last file in the device folder if there is
    if d_last_device:
        if os.listdir(d_last_device):
            last_device_img = d_last_device + '\\' + os.listdir(d_last_device)[-1]
    else:
        last_device_img = ''
    return last_device_img

def showimg_live(dir_date, device_name):
    # Show a file with cv2
    last_img = last_img_path(dir_date, device_name)
    # if there is an image already saved, show it
    if last_img:
        I = imread(last_img)
        I = cv2.resize(I, (960,540))
        I = cv2.applyColorMap(I, cv2.COLORMAP_HSV)
        cv2.imshow(device_name, I)

    while cv2.getWindowProperty(device_name, 0) >= 0:
        img_now = last_img_path(dir_date, device_name)
        if not filecmp.cmp(last_img, img_now):
            last_img = img_now
            I = imread(last_img)
            I = cv2.resize(I, (960,540))
            I = cv2.applyColorMap(I, cv2.COLORMAP_HSV)
            cv2.imshow(device_name, I)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return None

def main():
    dir_date = 'Z:\\data\\Undulator\\Y2020\\01-Jan\\20_0123'
    device_name = 'U_PhasicsFileCopy'

    # input
    # dir_date = input('Enter path of the scan date: ')
    # if not os.path.isdir(dir_date):
    #    print('Type ex.) Z:\\data\\Undulator\\Y2020\\01-Jan\\20_0121')
    #    raise SystemExit
    # device_name = input('Enter a device name. ex) U_PhasicsFileCopy: ')

    showimg_live(dir_date, device_name)

if __name__=='__main__':
    main()
