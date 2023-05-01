# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:44:28 2023

@author: ReiniervanMourik
"""

#%% init

from __future__ import annotations
from pathlib import Path

folder = Path(r"C:\Users\loasis\tau_systems\230420_undulator_analysis")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import NewType, TYPE_CHECKING

import sys
assert Path(sys.executable) == Path("C:/Users/loasis/tau_systems/230420_undulator_analysis/venv/Scripts/python.exe")

sys.path.append(r"C:\GEECS\Developers Version\source\GEECS-Plugins\dataanalysis-notebook\functions")
# a function that can properly import the 12-bit pngs that labview outputs
from pngTools import nBitPNG as load_bella_png

sys.path.append(r"C:\Users\loasis\tau_systems\github\BELLA_data_upload")
from htu_data import Scan
if TYPE_CHECKING:
    from htu_data import Shot

from progressbar import progressbar
from scipy.signal import convolve2d, fftconvolve

#%% functions


def clip_outliers(  img: np.ndarray, 
                    mad_multiplier: float = 5.2, 
                    nonzero_only: bool = True
                 ):
    """ Clips image values above a MAD-derived threshold.

    The threshold is median + mad_multiplier * MAD. 
    
    if nonzero_only (default True), the median and MAD only take nonzero values 
    into account, because otherwise in many pointing images the median and MAD are 
    simply 0.

    """
    
    def mad(x):
        """ Median absolute deviation
        """
        return np.median(np.abs(x - np.median(x)))

    
    img_outlier_clipped = img.copy()
    def get_threshold(X):
        return np.median(X) + mad_multiplier * mad(X)
    thresh = (get_threshold(img[img>1e-9]) if nonzero_only else get_threshold(img))
    img_outlier_clipped[img_outlier_clipped > thresh] = thresh
    
    return img_outlier_clipped


def load_image(scan: Scan, shot: int, device: str):
    return load_bella_png(
        scan.scan_folder/device/f"Scan{scan.scan:03d}_{device}_{shot:03d}.png"
    )


#%%

def analyze_undulatorrad2_images(scan: Scan, bkg_scan: Scan):
    for scan_parameter_value, scalar_data_group in scan.scalar_data_groupedby_scan_parameter_step():
        for (_, _, shot) in scalar_data_group.index:
            img = (
                load_bella_png(scan.scan_folder/"UC_UndulatorRad2"/f"Scan{scan.scan:03d}_UC_UndulatorRad2_{shot:03d}.png")
            )    

images = [load_bella_png(str(scan.scan_folder/"UC_UndulatorRad2"/f"Scan{scan.scan:03d}_UC_UndulatorRad2_{shot:03d}.png"))
          for shot in range(1,401)
         ]

scan = Scan('23_0411', 11)
images = []
for filepath in progressbar((scan.scan_folder/'UC_UndulatorRad2').iterdir(), max_value=sum(1 for _ in (scan.scan_folder/'UC_UndulatorRad2').iterdir())):
    if filepath.suffix == '.png':
        images.append(load_bella_png(str(filepath)))
        
from itertools import product



for img1, img2 in product(images[:10], images[:10]):
    C = fftconvolve(clip_outliers(img1), clip_outliers(img2)[::-1, ::-1], mode='same') / img1.size
    


#%%
import re

def load_images(run: str, scan: int):
    scan = Scan(run, scan)
    images = {}
    for filepath in progressbar((scan.scan_folder/'UC_UndulatorRad2').iterdir(), max_value=sum(1 for _ in (scan.scan_folder/'UC_UndulatorRad2').iterdir())):
        if filepath.suffix == '.png':
            shot = int(re.search(r"UC_UndulatorRad2_(\d{3}).png", filepath.name).group(1))
            images[shot] = clip_outliers(load_bella_png(str(filepath)).astype(float))

    return images

def plot_image(shot: int, p):
    # img = clip_outliers(
    #     load_bella_png(
    #         str(scan.scan_folder/'UC_UndulatorRad2'/f"Scan{scan.scan:03d}_UC_UndulatorRad2_{shot:03d}.png")
    #     )
    # )
    
    p.set_array(images[shot][::2,::2])
    t.set_text(f"shot{shot:03d}")

fig = plt.figure(f"{run}/Scan{scan_number:03d}")
fig.clf()
ax = fig.add_axes([0.125, 0.20, 0.775, 0.68])

p = ax.pcolormesh(clip_outliers(img1)[::2,::2])
t = ax.title

class Index:
    
    def __init__(self):
        self.shot = 1
    
    def next_image(self, event):
        if self.shot < 400:
            self.shot += 1
        plot_image(self.shot, p)
        plt.draw()
        
    def previous_image(self, event):
        if self.shot > 1:
            self.shot -= 1
        plot_image(self.shot, p)
        plt.draw()


#def undulatorrad2_viewer(run: str, scan_number: int):
scan = Scan(run, scan_number)


callback = Index()
ax_previous = fig.add_axes([0.4, 0.10, 0.05, 0.05])
ax_next = fig.add_axes([0.5, 0.10, 0.05, 0.05])
button_previous = Button(ax_previous, '<')
button_previous.on_clicked(callback.previous_image)
button_next =  Button(ax_next, '>')
button_next.on_clicked(callback.next_image)
plt.show()

#undulatorrad2_viewer('23_0411', 11)

#%% viewer for all undulator cameras
# 2023-04-20

from matplotlib.widgets import Button

device_list = (  [f'UC_ALineEbeam{i:d}' for i in range(1, 4)]
               + [f'UC_VisaEBeam{i:d}' for i in range(1, 10)]
               + ['UC_UndulatorRad2']
              )

fig = plt.figure('UndulatorViewer'); fig.clf()
fig, axs = plt.subplots(4, 4, num=fig.get_label(), subplot_kw={'aspect': 1.0})

device_plot_directory = {}
for device, ax in zip(device_list, axs.flatten()):
    ax.set_title(device)
    p = plt.pcolor(np.array([[]]))
    device_plot_directory[device] = {'ax': ax, 'p': p}

def plot_images(scan: Scan, shot: int):
    for device, ax_p in device_plot_directory.items():
        ax, p = ax_p['ax'], ax_p['p']
        image = clip_outliers(
            load_bella_png(
                scan.scan_folder/device/f"Scan{scan.scan:03d}_{device}_{shot:03d}.png"
            )
        )
        if p.get_array().shape == image[::3, ::3].shape:
            p.set_array(image[::3, ::3])
        else:
            ax.cla()
            ax.set_title(device)
            device_plot_directory[device]['p'] = ax.pcolormesh(image[::3, ::3])

    fig.suptitle(f"Scan{scan.scan:03d}/Shot{shot:03d}")        

class Index:
    
    def __init__(self, scan: Scan):
        self.scan = scan
        self.shot = 1
    
    def next_image(self, event):
        if self.shot < max(scan.scalar_data.index.get_level_values('Shotnumber')):
            self.shot += 1
        plot_images(self.scan, self.shot)
        plt.draw()
        
    def previous_image(self, event):
        if self.shot > 1:
            self.shot -= 1
        plot_images(self.scan, self.shot)
        plt.draw()


scan = Scan('23_0418', 59)


callback = Index(scan)
ax_previous = axs[3, 2]
ax_next = axs[3, 3]
button_previous = Button(ax_previous, '<')
button_previous.on_clicked(callback.previous_image)
button_next =  Button(ax_next, '>')
button_next.on_clicked(callback.next_image)
plt.show()




#%% define image sequences, ROIs, and determine background
# 2023-04-26

from progressbar import ProgressBar

DeviceName = NewType('DeviceName', str)
Array2D = NewType("Array2D", np.ndarray)

device_list = (  ['UC_ALineEbeam1', 'UC_ALineEBeam2', 'UC_ALineEBeam3',]
               + [f'UC_VisaEBeam{i:d}' for i in range(1, 10)]
               + ['UC_UndulatorRad2']
              )

image_sequences = [('EMQ Config 1: 1.2, -0.8 and 0.3 a', 
                    {device: Scan('23_0418', scan_number)
                     for device, scan_number 
                     in zip(device_list, [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 40])
                    },
                   ),
                   ('EMQ Config 1: 1.3, -0.9 0.3 (after aligna setpoint change)',
                       {device: Scan('23_0418', scan_number)
                        for device, scan_number 
                        in zip(device_list, [46, 47, 48,] + list(range(51, 60)) + [59])
                       },
                   )
                  ]

# in lower left x, lower left y, upper right x, upper right y
ROIs = {'UC_ALineEbeam1': [319, 204, 777, 701], 
        'UC_ALineEBeam2': [499, 261, 843, 656],
        'UC_ALineEBeam3': [256, 274, 528, 546],
        'UC_VisaEBeam1': [377, 0, 708, 252],
        'UC_VisaEBeam2': [100, 164, 434, 400],
        'UC_VisaEBeam3': [137, 185, 477, 469],
        'UC_VisaEBeam4': [192, 263, 541, 508],
        'UC_VisaEBeam5': [106, 167, 450, 427],
        'UC_VisaEBeam6': [147, 167, 500, 406],
        'UC_VisaEBeam7': [128, 140, 462, 490],
        'UC_VisaEBeam8': [111, 206, 401, 466],
        'UC_VisaEBeam9': [670, 341, 1141, 628],
        'UC_UndulatorRad2': [1420, 600, 2360, 1170],
       }

def crop_and_rotate(image: np.ndarray, device: str):
    x1, y1, x2, y2 = ROIs[device]
    cropped_image = image[y1:y2, x1:x2]
    
    if device in ['UC_ALineEbeam1', 'UC_ALineEBeam2']:
        # rotates the image counter-clockwise
        cropped_image = np.rot90(cropped_image)

    return cropped_image


def load_crop_and_rotate_image(shot: 'Shot', device: str):
    return crop_and_rotate(
               clip_outliers(
                   shot.images[device]['raw'].load()
               ), device
           )

# average image (w/o bkg-subtr) per sequence per device
average_images: list[tuple[str, dict[DeviceName, Array2D]]] = []
# images of elementwise minima, across all image_sequences, for background.
min_images: dict[DeviceName, Array2D] = {}

# count number of total shots, for progressbar
num_shots = 0
for sequence_comment, device_to_scan_map in image_sequences:
    for device, scan in device_to_scan_map.items():
        for _, shot in scan.shots.items():
            num_shots += 1

# determine minimum image (elementwise per device)
with ProgressBar(max_value=num_shots) as pb:
    for sequence_comment, device_to_scan_map in image_sequences:
        average_images_this_sequence = {}
        for device, scan in device_to_scan_map.items():
            sum_image = 0.0
            for  _, shot in scan.shots.items():
                image = load_crop_and_rotate_image(shot, device)
                sum_image += image
                if device in min_images:
                    min_images[device] = np.minimum(min_images[device], image)
                else:
                    min_images[device] = image
                pb.update(pb.value + 1)    
                
            average_images_this_sequence[device] = sum_image / len(scan.scalar_data)

        average_images.append((sequence_comment, average_images_this_sequence))

# plot minimum images
fig = plt.figure('min_images', figsize=(12, 12)); fig.clf();
fig, axs = plt.subplots(4, 4, num=fig.get_label(), subplot_kw={'aspect': 1.0})
for (device, min_image), ax in zip(min_images.items(), axs.flatten()):
    p = ax.pcolormesh(min_image)
    plt.colorbar(p, ax=ax)
    ax.set_title(device)

# plot average images by sequence
for sequence_comment, average_images_this_sequence in average_images:
    fig = plt.figure(sequence_comment, figsize=(12, 12)); fig.clf()
    fig, axs = plt.subplots(4, 4, num=fig.get_label(), subplot_kw={'aspect': 1.0})
    
    for device, ax in zip(device_list, axs.flatten()):
        ax.pcolormesh(average_images_this_sequence[device] - min_images[device])
        ax.set_title(device)
    
    fig.suptitle(sequence_comment)


#%%
scan = Scan('23_0418', 32)
images = []
for shot in scan.scalar_data.index.get_level_values('Shotnumber'):
    images.append( load_image(scan, shot, 'UC_VisaEBeam3'))
    
fig = plt.figure('UC_VisaEBeam3', figsize=(12, 12)); fig.clf();
fig, axs = plt.subplots(7, 8, num=fig.get_label(), subplot_kw={'aspect': 1.0})

for image, ax in zip(images, axs.flatten()):
    p = ax.pcolormesh(image - min_images['UC_VisaEBeam3'])
    plt.colorbar(p, ax=ax)
    p.set_clim(0, 1000)
    
scan = Scan('23_0418', 27)
images = []
for shot in scan.scalar_data.index.get_level_values('Shotnumber'):
    images.append( load_image(scan, shot, 'UC_ALineEbeam1'))
    
fig = plt.figure('UC_ALineEbeam1', figsize=(12, 12)); fig.clf();
fig, axs = plt.subplots(7, 8, num=fig.get_label(), subplot_kw={'aspect': 1.0})

for image, ax in zip(images, axs.flatten()):
    p = ax.pcolormesh(image - min_images['UC_ALineEbeam1'])
    plt.colorbar(p, ax=ax)
    p.set_clim(0, 3000)
    
            
scan = Scan('23_0418', 40)
images = []
for shot in scan.scalar_data.index.get_level_values('Shotnumber'):
    images.append( load_image(scan, shot, 'UC_VisaEBeam9'))
    
fig = plt.figure('UC_VisaEBeam9', figsize=(12, 12)); fig.clf();
fig, axs = plt.subplots(7, 8, num=fig.get_label(), subplot_kw={'aspect': 1.0})

for image, ax in zip(images, axs.flatten()):
    p = ax.pcolormesh(image - min_images['UC_VisaEBeam9'])
    plt.colorbar(p, ax=ax)
    p.set_clim(0, 400)
    
            
    
#%% calculate centroids and plot them.
# 2023-04-20

centroid_data = []

with ProgressBar(max_value=num_shots) as pb:
    for sequence_comment, device_to_scan_map in image_sequences:
        for device, scan in device_to_scan_map.items():
            for shot_number, shot in scan.shots.items():
                image: Array2D = load_crop_and_rotate_image(shot, device) - min_images[device]
                
                X, Y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
                image_sum: float = image.sum()
                centroid_x: float = np.sum(image * X) / image_sum
                centroid_y: float = np.sum(image * Y) / image_sum
                
                centroid_data.append((sequence_comment, device, shot, 
                                      centroid_x, centroid_y, image_sum
                                     )
                                    )

                pb.increment()

centroid_df = pd.DataFrame(centroid_data, columns=['sequence', 'device', 'shot', 'centroid_x', 'centroid_y', 'image_sum'])\
                  .set_index(['sequence', 'device', 'shot'])


fig = plt.figure('centroids', figsize=(12, 12)); fig.clf();
fig, axs = plt.subplots(4, 4, num=fig.get_label(), subplot_kw={'aspect': 1.0})

for device, ax in zip(device_list, axs.flatten()):
    device_group = centroid_df.groupby('device').get_group(device)
    for sequence, sequence_group in device_group.groupby('sequence'):
        ax.scatter(sequence_group['centroid_x'], sequence_group['centroid_y'], 
                   s=sequence_group['image_sum'] / device_group['image_sum'].median() * 10,
                   label=sequence, alpha=0.1
                  )
    
    ax.set_title(device)
    x1, y1, x2, y2 = ROIs[device]
    ax.set_xlim(0, x2 - x1)
    ax.set_ylim(0, y2 - y1)

axs[0, 0].legend()



