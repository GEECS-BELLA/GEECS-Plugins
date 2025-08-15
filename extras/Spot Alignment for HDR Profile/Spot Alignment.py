#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:47:24 2023

@author: Anthony Valente Vazquez, UC Berkeley
"""

import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
from skimage import io
from scipy.signal import savgol_filter
import sys
from tkinter import Tk, filedialog
import time
import numpngw
import ni_vision


class Alignment:

    # initialize folder and images to be used
    def __init__(self):
        self.processed = None
        self.num_folders = 0
        self.folder_names = []
        self.current_path = Path(os.getcwd())
        self.root = Tk()
        self.root.withdraw()
        self.roi_paths_and_coords = {}
        self.folder_names = []
        self.bkg_folders = []
        self.test_scores = {}
        self.best_match = {}
        self.align_and_process_called = False
        self.final_save = False
        self.align_all = False
        self.crop = False
        self.next_roi_call = False


    # troubleshooting
    def debug(self):
        # print(self.current_path)
        # print(self.DragAndDrop)
        print("done")

    def select_bkg_file(self):
        self.root.update()
        file_path = filedialog.askopenfilename()
        return file_path
        # Determine the operating system

    def select_folder(self):
        self.root.update()
        folder_path = filedialog.askdirectory()
        return folder_path if folder_path else None

    ###############################################################################

    # user input for num and names of folders containing images
    def ask_for_folders(self):
        print("--------------------------------------------------------------------------------------------\n"
              "Hello! Welcome to my High Accuracy Beam Identification and Optimized Alignment program :D\n"
              "--------------------------------------------------------------------------------------------\n"
              "\n---Please order folders of images from least saturated to most saturated.---\n")
        while True:
            try:
                self.num_folders = int(input("Enter the number of folders you have for me: "))
                break  # Exit the loop if the input is an integer
            except ValueError:
                print("Invalid input. Please enter an integer.")
        time.sleep(0.5)
        for i in range(self.num_folders):
            print("\nSelect Folder:")
            time.sleep(0.5)
            folder_path = self.select_folder()
            if folder_path:
                self.folder_names.append(Path(folder_path))
        print("\nFolders Selected:")
        for folder in self.folder_names:
            print(folder)

    # see if user has a background image that they would prefer using instead of a predetermined array
    def ask_for_bkg_image(self):
        bkg_image = input("\nDo you have a background image for me to use? (y/n): ")
        self.bkg_image_answer = bkg_image
        time.sleep(0.5)
        if bkg_image.lower() == 'y':
            file_path = self.select_bkg_file()
            if file_path:
                self.bkg_image = Path(file_path)
                print("awesome :D")
                time.sleep(0.5)
                print(f"File path selected:\n{self.bkg_image}")
            else:
                print("No file selected.")
                self.bkg_image = None
        else:
            time.sleep(0.5)
            print("Okay...I will use my own estimation...\n")
            time.sleep(0.5)
            self.bkg_image = None

    def ask_best_or_all(self):
        time.sleep(0.5)
        print("Would you like to find the best matches and process or align/process all of your images?")
        print("------------------------------------------------------")
        while True:
            self.decision = input("best = 1, all = 2: ")
            if self.decision in ['1', '2']:
                self.decision = int(self.decision)
                break
            else:
                print("Invalid input! Please enter either '1' or '2'.")
        if self.decision == 2:
            self.decision = int(self.decision)
            self.sub_bkg = "Aligned Images"
            self.aligned_path = self.current_path / self.sub_bkg
            self.bkg_path = self.aligned_path
            # print(self.bkg_path)
            self.bkg_folders.append(self.aligned_path)
            if not os.path.exists(self.aligned_path):
                os.mkdir(self.aligned_path)
            else:
                # Remove existing images in the folder
                for filename in os.listdir(self.aligned_path):
                    os.remove(os.path.join(self.aligned_path, filename))
        else:
            pass
    # take ROI of first image to be used to find coords of rest of their images
    def initial_ROIs(self):
        # print("\nFolder Selection:")
        # print("--------------------------------------")
        # for i, item in enumerate(self.bkg_folders, start=1):
        #     print(f"{i}. {item}")
        # print("--------------------------------------")
        while True:
            try:
                # selected_folder_index = int(input("\nWhich folder do you want your ROI to come from? \n"
                #                                   "Type the folder number from the list above: "))
                selected_folder_index = int(1)
                if 1 <= selected_folder_index <= len(self.bkg_folders):
                    break  # Exit the loop if the input is within the valid range
                else:
                    print("Invalid folder number. Please select a valid folder number.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        self.first_ROI = self.bkg_folders[selected_folder_index - 1]
        time.sleep(0.5)
        print("\nPlease set ROI... \n"
              "\n***To restart ROI, simply drag new box over old one, confirm and close ROI window by pressing Enter Key twice.")
        self.first_ROI = next(file for file in self.first_ROI.iterdir() if file.is_file() and file.suffix == ".png")
        self.first_roi_path = os.path.abspath(os.path.join(self.first_ROI, os.pardir))
        self.first_ROI_filename = os.path.basename(self.first_ROI)
        self.find_ROI()

    def single_ROI(self):
        for i, folder in enumerate(self.folder_names):
            for filename in os.listdir(folder):
                if filename.startswith("ROI"):
                    file_path = os.path.join(folder, filename)
                    os.remove(file_path)
        while True:
            try:
                selected_folder_index = int(1)
                if 1 <= selected_folder_index <= len(self.bkg_folders):
                    break  # Exit the loop if the input is within the valid range
                else:
                    print("Invalid folder number. Please select a valid folder number.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        self.first_ROI = self.folder_names[selected_folder_index - 1]
        time.sleep(0.5)
        print("\nPlease set ROI... \n"
              "\n***To restart ROI, simply drag new box over old one, confirm and close ROI window by pressing Enter Key twice.")
        self.first_ROI = next(file for file in self.first_ROI.iterdir() if file.is_file() and file.suffix == ".png")
        self.first_roi_path = os.path.abspath(os.path.join(self.first_ROI, os.pardir))
        self.first_ROI_filename = os.path.basename(self.first_ROI)
        self.find_ROI()

    ###############################################################################
    def find_remaining_ROIs(self):
        #purpose of this funciton is to get the dictionary with roi filenames and coords for cropping
        time.sleep(0.5)
        print("Okay, I will locate ROIs of first folder for best match search:")
        time.sleep(1)
        folder = self.directory
        initial_template = self.new_image_path
        file_list = []
        for filename in os.listdir(folder):
            full_path = os.path.join(folder, filename)
            if os.path.isfile(full_path) and not filename.startswith('ROI') and filename.lower().endswith('.png'):
                file_list.append(full_path)
        # self.find_rois = True
        self.ROI_results(file_list, initial_template)
        # self.find_rois = False
        os.remove(initial_template)
        self.roi_paths = file_list
        # for file_path, coords in self.roi_paths_and_coords.items():
        #     print(f"{file_path}: {coords}")

    def subtract_bkg_from_selected_folders(self):
        print("subtracting background of files...")
        print("-------------------------------")
        time.sleep(0.5)
        self.sub_bkg = "Removed_background"

        for i, folder in enumerate(self.folder_names):
            self.bkg_path = folder / self.sub_bkg
            # print(self.bkg_path)
            self.bkg_folders.append(self.bkg_path)
            if not os.path.exists(self.bkg_path):
                os.mkdir(self.bkg_path)
            else:
                # Remove existing images in the folder
                for filename in os.listdir(self.bkg_path):
                    os.remove(os.path.join(self.bkg_path, filename))
            image_list = []
            for file in os.listdir(folder):
                if file.endswith('.png'):
                    self.pre_file = file
                    image_path = os.path.join(folder, file)
                    image_list.append(image_path)
            image_list.sort()
            self.folder = folder
            if self.bkg_image:
                bkg_image = self.bkg_image
                # self.image_rmv_bkg(image_list, bkg_image)
                self.image_rmv_bkg_for_PM(image_list, bkg_image)
            else:
                # self.default_rmv_bkg(image_list)
                self.default_rmv_bkg_for_PM(image_list)
            if i+1 != len(self.folder_names):
                print("next folder...")
            elif i+1 == len(self.folder_names):
                print("done with folders")
                print("-------------------------------")
            else:
                print("what happened...?")


    def find_ROI(self):
        image = cv2.imread(str(self.first_ROI))
        filename = os.path.splitext(os.path.basename(str(self.first_ROI)))[0]
        # Display the image and select ROI
        bbox = cv2.selectROI('Select ROI', image, fromCenter=False, showCrosshair=True)
        # Extract the coordinates of the ROI
        x, y, w, h = bbox
        # Save the coordinates
        coords = (x, y, x + w, y + h)
        # Print the coordinates
        print("ROI coordinates:", coords, "\n")
        # # Crop the ROI from the image
        roi = image[y:y + h, x:x + w]
        # Extract the directory path from image_path
        self.directory = self.first_roi_path
        # Construct the new file path
        self.new_image_path = os.path.join(self.directory, "ROI_" + filename +'.png')
        # Save the cropped image in the same folder as image_path
        cv2.imwrite(self.new_image_path, roi)
        # # Display the cropped ROI
        # cv2.imshow('ROI', roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def find_best_results(self):
        time.sleep(0.5)
        folders = self.bkg_folders
        first_run = folders[1:2]
        # print(folders)
        first_folder = sorted(self.roi_paths)

        print("ROIs have been found...will save ROIs to list for best match test")
        rois = []
        # saves cropped images as templates for matchTemplate
        for image in first_folder:
            file_paths = []
            template = cv2.imread(image)
            filename = os.path.splitext(os.path.basename(str(image)))[0]
            roi_path = os.path.join(self.directory, "ROI_" + filename + '.png')
            x, y, x2, y2 = self.roi_paths_and_coords[image]
            # print(roi)
            crop = template[y:y2, x:x2]
            cv2.imwrite(roi_path, crop)
            rois.append(roi_path)
        time.sleep(0.5)
        print("Done saving...")
        time.sleep(0.5)

        best_results = []
        self.all_files = []
        next_roi = []
        print("-----------------------")
        for roi_filename in rois:
            print(f"testing {os.path.basename(roi_filename)}")
            for folder in first_run:
                # file_list = os.listdir(folder)
                file_paths = sorted([os.path.join(folder, x) for x in os.listdir(folder) if x.lower().endswith(".png")])
                # print("file paths", file_paths)
                # print("roi filename", roi_filename)
                self.ROI_results(file_paths, roi_filename)
                best_results.append(self.highest_element[0])
                # print()

                if len(best_results) == len(rois):
                    # for item in best_results:
                    #     print(item)
                    highest_max_val = max(best_results, key=lambda x: x[2])[2]  # Find the highest max-val from best_results
                    best_results = [result for result in best_results if
                                    result[2] == highest_max_val]  # Keep only highest max-val results
                    # print("should only have one element:\n", best_results)
                    self.all_files.append(best_results[0])

                    # self.all_files.append(self.max_results[0]
                    # for item in best_results:
                    #     print("all best results:", item)
                else:
                    pass
        print("-----------------------")
        # this will append the first ROI max result with the whole first folder
        first_temp = best_results[0][0]
        self.first_temp = first_temp
        directory, file_with_path = os.path.split(first_temp)
        # Remove "ROI_" from the filename
        new_filename = file_with_path.replace("ROI_", "")
        # Combine the directory path and the modified filename to get the new full path
        new_full_path = [os.path.join(directory, new_filename)]
        # print(new_full_path)
        self.ROI_results(new_full_path, first_temp)
        # print("this shoudl be the first temp as an image", self.max_results)
        first_result = self.highest_element[0]
        self.all_files.append(first_result)
        # print("now there should be two elements:")
        # for item in self.all_files:
        #     print(item)
        #if this condition is satisfied, you only need to align 2 images, otherwise continue and align 3+ images
        reorder = []
        reorder.append(self.all_files[1])
        reorder.append(self.all_files[0])
        # for item in reorder:
        #     print(item)
        time.sleep(0.5)

        if len(folders) >= 3:
            remaining_test_folders = folders[2:]
            filepath = reorder[1][1]
            original_filepath = filepath
            separator = os.path.sep
            # Split the path into components
            path_components = original_filepath.split(separator)
            # Find and remove the 'Removed_background' component
            index_to_remove = path_components.index('Removed_background')
            path_components.pop(index_to_remove)
            # Reconstruct the updated path and add '.png' extension
            original_filepath = separator.join(path_components)
            coords = (reorder[1][5], reorder[1][6], reorder[1][7], reorder[1][8])
            if self.bkg_image_answer.lower() == 'y':
                self.next_roi_call = True
                # self.image_rmv_bkg([original_filepath])
                bkg_image = self.bkg_image
                self.image_rmv_bkg_for_PM([original_filepath], bkg_image)
                new_roiimage = self.new_roiimage
                self.next_roi_call = False
            else:
                self.next_roi_call = True
                # self.default_rmv_bkg([original_filepath])
                self.default_rmv_bkg_for_PM([original_filepath])
                new_roiimage = self.new_roiimage
                self.next_roi_call = False
            # Crop the image based on coordinates
            x1, y1, x2, y2 = coords
            cropped_image = new_roiimage[y1:y2, x1:x2]
            # Generate the new filename
            base_filename = os.path.splitext(os.path.basename(filepath))[0]  # Get filename without extension
            new_filename = 'ROI_' + base_filename
            # Save the cropped image using OpenCV with the new filename
            new_roi_filepath = os.path.join(os.path.dirname(filepath), new_filename)
            # print(new_roi_filepath)
            new_roi_filepath = str(new_roi_filepath) + ".png"
            img = Image.fromarray(cropped_image)
            img.save(new_roi_filepath)
            # self.write_binary(cropped_image, new_roi_filepath)


            for folder in remaining_test_folders:
                file_list = os.listdir(folder)
                file_paths = sorted([os.path.join(folder, x) for x in file_list if x.lower().endswith(".png")])
                # Use the highest_max_val_file_paths obtained from the first run
                self.ROI_results(file_paths, new_roi_filepath)
                best_result = self.max_results[0]  # Since we are aligning only one ROI with multiple images in each folder
                # print(best_result)
                reorder.append(best_result)

                #create new ROI for next folder
                filepath = self.max_results[0][1]
                original_filepath = filepath
                separator = os.path.sep
                # Split the path into components
                path_components = original_filepath.split(separator)
                # Find and remove the 'Removed_background' component
                index_to_remove = path_components.index('Removed_background')
                path_components.pop(index_to_remove)
                # Reconstruct the updated path and add '.png' extension
                original_filepath = separator.join(path_components)
                coords = (self.max_results[0][5], self.max_results[0][6], self.max_results[0][7], self.max_results[0][8])
                if self.bkg_image_answer.lower() == 'y':
                    self.next_roi_call = True
                    # self.image_rmv_bkg([original_filepath])
                    bkg_image = self.bkg_image
                    self.image_rmv_bkg_for_PM([original_filepath], bkg_image)
                    new_roiimage = self.new_roiimage
                    self.next_roi_call = False
                else:
                    self.next_roi_call = True
                    # self.default_rmv_bkg([original_filepath])
                    self.default_rmv_bkg_for_PM([original_filepath])
                    new_roiimage = self.new_roiimage
                    self.next_roi_call = False
                # Crop the image based on coordinates
                x1, y1, x2, y2 = coords
                cropped_image = new_roiimage[y1:y2, x1:x2]
                # Generate the new filename
                base_filename = os.path.splitext(os.path.basename(filepath))[0]  # Get filename without extension
                new_filename = 'ROI_' + base_filename
                # Save the cropped image using OpenCV with the new filename
                new_roi_filepath = os.path.join(os.path.dirname(filepath), new_filename)
                # print(new_roi_filepath)
                new_roi_filepath = str(new_roi_filepath) + ".png"
                img = Image.fromarray(cropped_image)
                img.save(new_roi_filepath)
            time.sleep(0.5)
            self.reorder = reorder
            print("all best results:")
            print("-----------------------")
            for item in reorder:
                print(os.path.basename(item[0]), os.path.basename(item[0]))
            print("-----------------------\n")
            time.sleep(0.5)
            # print("\n")
        else:
            self.reorder = reorder
            print("all best results:")
            print("-----------------------")
            for item in reorder:
                print(os.path.basename(item[0]), os.path.basename(item[0]))
            print("-----------------------")
            time.sleep(0.5)
            # print("\n")

    #####################################################################################

    def align_everything(self):
        initial_template = self.new_image_path
        self.align_all = True
        print(initial_template)
        image_folders = self.folder_names
        for folder in image_folders:
            original_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]
            self.align_and_process_called = True
            self.final_image_arrays = {}
            if self.bkg_image == None:
                # initial_template = self.new_image_path
                self.default_value = True
                self.default_rmv_bkg([initial_template])
                # self.default_rmv_bkg_for_PM([initial_template])
                original_files = [path for path in original_files if "ROI_" not in path]
                # print(original_files)
                self.default_rmv_bkg(original_files)
                # self.default_rmv_bkg_for_PM(original_files)
                print("using default")
            else:
                # initial_template = self.new_image_path
                bkg_image = self.bkg_image
                self.image_value = True
                self.crop = True
                self.image_rmv_bkg([initial_template], bkg_image)
                original_files = [path for path in original_files if "ROI_" not in path]
                self.crop = False
                # print(original_files)
                self.image_rmv_bkg(original_files, bkg_image)
                print("using image")
            # print(self.final_image_arrays)

            time.sleep(0.5)
            print("--------------------------------")
            for filename, array in self.final_image_arrays.items():
                print(f"aligning {os.path.basename(filename)}...")
                image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                mask = cv2.inRange(image, 0, 254)  # 80%
                template = cv2.imread(initial_template, cv2.IMREAD_GRAYSCALE)
                w = template.shape[1]
                h = template.shape[0]
                mask = cv2.resize(mask, (w, h))
                result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED, mask=mask)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                top_left = max_loc
                coords = (top_left[0], top_left[1], top_left[0] + w, top_left[1] + h)
                result_center_x = int(top_left[0] + w / 2)
                result_center_y = int(top_left[1] + h / 2)
                # Translate the center of the result to the center of the image
                image_center_x = int(image.shape[1] / 2)
                image_center_y = int(image.shape[0] / 2)
                dx = int(image_center_x - result_center_x)
                dy = int(image_center_y - result_center_y)

                M = np.float32([[1, 0, dx], [0, 1, dy]])
                translated_image = cv2.warpAffine(array, M, (array.shape[1], array.shape[0]))
                original_width, original_height = translated_image.shape

                original_area = original_width * original_height
                cropped_side = int((original_area * 0.35) ** 0.5)
                # print(cropped_side)
                starting_y = (original_width - cropped_side) // 2
                starting_x = (original_height - cropped_side) // 2
                # print(starting_x, starting_y)
                # print(starting_x + cropped_side, starting_y + cropped_side)
                cropped_image = translated_image[starting_y:starting_y + cropped_side, starting_x:starting_x + cropped_side]
                saving_path = str(self.aligned_path / os.path.splitext(os.path.basename(filename))[0])
                # EF_loop.append((saving_path, cropped_image))
                self.write_binary(cropped_image, saving_path)
            file_list = os.listdir(self.aligned_path)

            # Loop through the files and delete those that start with "ROI"
            for filename in file_list:
                if filename.startswith("ROI"):
                    file_path = os.path.join(self.aligned_path, filename)
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
            print("all images have been aligned")
            print("--------------------")
        print("done.")

#######################################################
    def image_rmv_bkg(self, image_list, bkg_image):
        if self.align_all == True:
            background = ni_vision.read_imaq_image(bkg_image)
            background = background.astype(np.float64)
            target_shape = ni_vision.read_imaq_image(image_list[0])
            target_shape = target_shape.astype(np.float64)
            target_height, target_width = target_shape.shape
            # Crop the first image to match the size of the target image
            background = background[:target_height, :target_width]
        else:
            background = ni_vision.read_imaq_image(bkg_image)
            background = background.astype(np.float64)

        for i in range(len(image_list)):
            bkg_sub_img = ni_vision.read_imaq_image(image_list[i])
            bkg_sub_img = bkg_sub_img.astype(np.float64)
            filename = os.path.splitext(os.path.basename(image_list[i]))[0]
            print(f"removing {filename}'s background")

            max_count = np.max(bkg_sub_img)
            # Subtract background directly using the imported 'bkg_img'
            bkg_sub_img = (bkg_sub_img - background)
            print(bkg_sub_img.min(), bkg_sub_img.max())
            # Subtract background
            bkg_sub_img[np.where(bkg_sub_img > max_count)] = max_count
            bkg_sub_img = bkg_sub_img.astype(np.float32)
            # Use the original filename as the base name for the binary file
            sub_bkg_filename = f'{filename}'
            if self.align_all != True:
                sub_bkg_filepath = os.path.join(self.bkg_path, sub_bkg_filename)
            else:
                sub_bkg_filepath = os.path.join(self.aligned_path, sub_bkg_filename)
            if self.align_and_process_called == True:
                # Execute the if statement and process the background removal as before
                if self.image_value == True:
                    self.bkg_sub_img = bkg_sub_img
                    sub_bkg_filepath = image_list[i]
                    self.sub_bkg_filepath = sub_bkg_filepath
                    self.final_image_arrays[self.sub_bkg_filepath] = self.bkg_sub_img
                else:
                    pass
            else:
                # Call write_binary with bkg_sub_img and the constructed binary filepath
                if self.next_roi_call == True:
                    self.new_roiimage = bkg_sub_img
                else:
                    self.write_binary(bkg_sub_img, sub_bkg_filepath)

    def default_rmv_bkg(self, image_list):
        # need to add a folder for removed background images
        for i in range(len(image_list)):
            bkg_sub_img = ni_vision.read_imaq_image(image_list[i])
            bkg_sub_img = bkg_sub_img.astype(np.float64)
            filename = os.path.splitext(os.path.basename(image_list[i]))[0]
            print(f"removing {filename}'s background")
            max_count = np.max(bkg_sub_img)
            n_bottom_rows = -10
            bkg_avg = np.mean(bkg_sub_img[n_bottom_rows:, :], axis=0)
            window_length = max(int(len(bkg_avg) / 3), 2)
            if window_length % 2 == 0:
                window_length += 1

            bkg = savgol_filter(bkg_avg, window_length, 3)
            bkg_sub_img = (bkg_sub_img - bkg)
            # Subtract background
            bkg_sub_img[np.where(bkg_sub_img > max_count)] = max_count
            bkg_sub_img = bkg_sub_img.astype(np.float32)
            # Use the original filename as the base name for the binary file
            sub_bkg_filename = f'{filename}'
            if self.align_all != True:
                sub_bkg_filepath = os.path.join(self.bkg_path, sub_bkg_filename)
            else:
                sub_bkg_filepath = os.path.join(self.aligned_path, sub_bkg_filename)
            # Check if align_and_process method has been called before executing the if statement
            if self.align_and_process_called == True:
                # Execute the if statement and process the background removal as before
                if self.default_value == True:
                    self.bkg_sub_img = bkg_sub_img
                    sub_bkg_filepath = image_list[i]
                    self.sub_bkg_filepath = sub_bkg_filepath
                    self.final_image_arrays[self.sub_bkg_filepath] = self.bkg_sub_img
                else:
                    pass
            else:
                # Call write_binary with bkg_sub_img and the constructed binary filepath
                if self.next_roi_call ==True:
                    self.new_roiimage = bkg_sub_img
                else:
                    self.write_binary(bkg_sub_img, sub_bkg_filepath)
#######################################################

    def image_rmv_bkg_for_PM(self, image_list, bkg_image):
        if self.align_all == True or self.next_roi_call == True:
            background = io.imread(bkg_image)
            # background = background.astype(np.float64)
            target_shape = io.imread(image_list[0])
            # target_shape = target_shape.astype(np.float64)
            target_height, target_width = target_shape.shape
            # Crop the first image to match the size of the target image
            background = background[:target_height, :target_width]
            max_count = 2 ** 16 - 1
            background = np.real(np.log10(background))
            background[np.isnan(background)] = 0.
            background[np.isinf(background) & (background < 0)] = 0.
            background[np.isinf(background) & (background > 0)] = max_count
        else:
            background = io.imread(bkg_image)
            # background = background.astype(np.float64)
            max_count = 2 ** 16 - 1
            background = np.real(np.log10(background))
            background[np.isnan(background)] = 0.
            background[np.isinf(background) & (background < 0)] = 0.
            background[np.isinf(background) & (background > 0)] = max_count

        for i in range(len(image_list)):
            bkg_sub_img = io.imread(image_list[i])
            # bkg_sub_img = bkg_sub_img.astype(np.float64)
            filename = os.path.basename(image_list[i])
            print(f"removing {filename}'s background")
            max_count = 2 ** 16 - 1
            bkg_sub_img = np.real(np.log10(bkg_sub_img))
            bkg_sub_img[np.isnan(bkg_sub_img)] = 0.
            bkg_sub_img[np.isinf(bkg_sub_img) & (bkg_sub_img < 0)] = 0.
            bkg_sub_img[np.isinf(bkg_sub_img) & (bkg_sub_img > 0)] = max_count

            # Subtract background directly using the imported 'bkg_img'
            bkg_sub_img = (bkg_sub_img - background)
            bkg_sub_img[np.where(bkg_sub_img < 0)] = 0.
            # print(bkg_sub_img.min(), bkg_sub_img.max())
            # Subtract background
            bkg_sub_img = max_count / (np.max(bkg_sub_img) - np.min(bkg_sub_img)) * (bkg_sub_img - np.min(bkg_sub_img))
            bkg_sub_img *= 1
            bkg_sub_img[np.where(bkg_sub_img > max_count)] = max_count
            bkg_sub_img = bkg_sub_img.astype(np.int32)

            # Use the original filename as the base name for the binary file
            # sub_bkg_filename = f'{filename}'
            if self.align_all != True:
                sub_bkg_filepath = os.path.join(self.bkg_path, filename)
            else:
                sub_bkg_filepath = os.path.join(self.aligned_path, filename)
            if self.next_roi_call == True:
                self.new_roiimage = bkg_sub_img
            else:
                img = Image.fromarray(bkg_sub_img)
                img.save(sub_bkg_filepath)

    def default_rmv_bkg_for_PM(self, image_list):

        for i in range(len(image_list)):
            bkg_sub_img = io.imread(image_list[i])
            # bkg_sub_img = bkg_sub_img.astype(np.float64)
            filename = os.path.basename(image_list[i])
            print(f"removing {filename}'s background")
            # max_count = np.max(bkg_sub_img)

            max_count = 2 ** 16 - 1
            bkg_sub_img = np.real(np.log10(bkg_sub_img))
            bkg_sub_img[np.isnan(bkg_sub_img)] = 0.
            bkg_sub_img[np.isinf(bkg_sub_img) & (bkg_sub_img < 0)] = 0.
            bkg_sub_img[np.isinf(bkg_sub_img) & (bkg_sub_img > 0)] = max_count

            n_bottom_rows = -60
            bkg_avg = np.mean(bkg_sub_img[n_bottom_rows:, :], axis=0)
            window_length = max(int(len(bkg_avg) / 3), 2)
            if window_length % 2 == 0:
                window_length += 1

            bkg = savgol_filter(bkg_avg, window_length, 3)
            bkg_sub_img = (bkg_sub_img - bkg)
            # Subtract background

            bkg_sub_img[np.where(bkg_sub_img < 0)] = 0.

            # Normalize and rescale
            bkg_sub_img = max_count / (np.max(bkg_sub_img) - np.min(bkg_sub_img)) * (bkg_sub_img - np.min(bkg_sub_img))
            bkg_sub_img *= 1
            bkg_sub_img[np.where(bkg_sub_img > max_count)] = max_count
            bkg_sub_img[np.where(bkg_sub_img >= 0.8 * max_count)] = 0
            bkg_sub_img = bkg_sub_img.astype(np.int32)
            # Use the original filename as the base name for the binary file
            # sub_bkg_filename = f'{filename}'
            if self.align_all != True:
                sub_bkg_filepath = os.path.join(self.bkg_path, filename)
            else:
                sub_bkg_filepath = os.path.join(self.aligned_path, filename)
            if self.next_roi_call == True:
                self.new_roiimage = bkg_sub_img
            else:
                img = Image.fromarray(bkg_sub_img)
                img.save(sub_bkg_filepath)

#############
    def ROI_results(self, folder, initial_template):
        self.max_results = []
        self.results = []
        for file_path in folder:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.inRange(image, 0, 204)  # 80%
            template = cv2.imread(initial_template, cv2.IMREAD_GRAYSCALE)
            w = template.shape[1]
            h = template.shape[0]
            mask = cv2.resize(mask, (w, h))
            result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED, mask=mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            coords = (top_left[0], top_left[1], top_left[0] + w, top_left[1] + h)
            result_center_x = int(top_left[0] + w / 2)
            result_center_y = int(top_left[1] + h / 2)
            # Translate the center of the result to the center of the image
            image_center_x = int(image.shape[1] / 2)
            image_center_y = int(image.shape[0] / 2)
            dx = int(image_center_x - result_center_x)
            dy = int(image_center_y - result_center_y)

            self.roi_paths_and_coords[file_path] = coords  # Append to the dictionary
            self.coords = coords

            self.max_results.append((initial_template, file_path, max_val, dx, dy, coords[0], coords[1], coords[2], coords[3]))
        self.first_folder = self.max_results
        # print("all results for this tempalte:")
        # for item in self.first_folder:
        #     print(item)
        if self.find_best_results:
            highest_max_val = max(self.max_results, key=lambda x: x[2])[2]
            highest_element = [(template, file_path, max_val, dx, dy, TL0, TL1, BR0, BR1) for template, file_path, max_val, dx, dy, TL0, TL1, BR0, BR1 in self.max_results if
                               max_val == highest_max_val]
            # print("these are the results \n",self.max_results)
            self.highest_element = highest_element
            # print(self.max_results)
        else:
            pass
    def align_best(self):
        time.sleep(0.5)
        if self.decision == 1:
            if self.decision == 1:
                #get dict of names of template ROIS and their coords
                self.find_remaining_ROIs()
                first_folder_values = self.results
                self.find_best_results()

                #here the first folder has the name of test image, result image, result, and (dx,dy) to center
                #good up to here, dict is made with scans of first folder and their ROI coords
                list_of_data = self.reorder
                # print(list_of_data)
                list_of_data = [
                    (t[0], os.path.join(os.path.dirname(os.path.split(t[1])[0]), os.path.split(t[1])[1]), t[2], t[3], t[4], t[5], t[6], t[7], t[8])
                    for t in list_of_data
                ]

                temp = self.first_temp

                original_files = [item[1] for item in list_of_data]
                # for item in original_files:
                #     print(item)
                self.processed = "Best images"
                self.best_folder = self.current_path / self.processed
                # print(self.bkg_path)
                self.bkg_folders.append(self.best_folder)
                if not os.path.exists(self.best_folder):
                    os.mkdir(self.best_folder)
                else:
                    # Remove existing images in the folder
                    for filename in os.listdir(self.best_folder):
                        os.remove(os.path.join(self.best_folder, filename))
                # os.chdir(self.best_folder)
                time.sleep(1)
                print("I will now begin processing:")
                print("------------------------------")
                self.align_and_process(original_files, list_of_data)
        else:
            pass

    # Read/Write functions
    def align_and_process(self, original_files, list_of_data):
        self.align_and_process_called = True
        self.final_image_arrays = {}
        if self.bkg_image == None:
            self.default_value = True
            self.default_rmv_bkg(original_files)
            # print("using default")
        else:
            self.image_value = True
            print(original_files)
            bkg_image = self.bkg_image
            self.image_rmv_bkg(original_files, bkg_image)
            # print("using image")
        print("--------------------------------------")
        # for filename in self.final_image_arrays.keys():
        #     print("final images to be used:", filename)
        # print("--------------------------------------")
        time.sleep(0.5)
        print("\nBeginning centering...\n")
        EF_loop = []
        for filename, array in self.final_image_arrays.items():
            for tuple_item in list_of_data:
                if any(filename in str(element) for element in tuple_item):
                    dx = tuple_item[3]  # Third element (index 2) is dx
                    dy = tuple_item[4]
                    # print("this is dx and dy:\n", dx,dy)
                    break
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            translated_image = cv2.warpAffine(array, M, (array.shape[1], array.shape[0]))
            original_width, original_height = translated_image.shape

            original_area = original_width * original_height
            cropped_side = int((original_area * 0.35) ** 0.5)
            # print(cropped_side)
            starting_y = (original_width - cropped_side) // 2
            starting_x = (original_height - cropped_side) // 2
            # print(starting_x, starting_y)
            # print(starting_x + cropped_side, starting_y + cropped_side)
            cropped_image = translated_image[starting_y:starting_y + cropped_side, starting_x:starting_x + cropped_side]
            saving_path = str(self.best_folder / os.path.splitext(os.path.basename(filename))[0])
            EF_loop.append((saving_path, cropped_image))
            self.write_binary(cropped_image, saving_path)
        for item in EF_loop:
            print(item[0])
        time.sleep(0.5)
        print("-----------------------------")
        print(f"\nimages will be saved here: {os.path.dirname(EF_loop[0][0])}")


        n = len(EF_loop)
        print("***if using Tony's HDR Script found at D:\\Server\\10_Software\\ajg builds\\LaserFocusHDR, these numbers are important..."
              "\n*** otherwise use HDR Exposure Time.py to get HDR image")
        for i in range(len(EF_loop)):
            unsat = EF_loop[i][1]
            if i + 1 >= len(EF_loop):
                break
            else:
                sat_region = EF_loop[i+1][1]
                upper_thresh = 0.47 * np.max(EF_loop[i+1][1])
                lower_thresh = 0.23 * np.max(EF_loop[i+1][1])
                filtered_sat = sat_region[(sat_region >= lower_thresh) & (sat_region <= upper_thresh)]
                # Get the pixel positions of the filtered values
                pixel_positions_sat = np.where((sat_region >= lower_thresh) & (sat_region <= upper_thresh))
                filtered_unsat = unsat[pixel_positions_sat]
                exposure_factor = np.sum(filtered_sat) / np.sum(filtered_unsat)
                print(f"\nthis is the exposure factor for\n {os.path.basename(EF_loop[i+1][0])}: {exposure_factor}")
        print(f"\nthis is the exposure factor for\n {os.path.basename(EF_loop[0][0])}: 1.0 ")
        print("-----------------------------")
        time.sleep(0.5)
        print("done.")

    def write_binary(self, data, bin_filepath):
        data_int = (65535 * ((data - np.min(data)) / np.ptp(data))).astype(np.uint16)
        lines = ['[Scaling]', 'min = %f' % np.min(data), 'max = %f' % np.max(data)]

        # Update the bin_filepath to point to the subtracted background folder
        bin_filepath = os.path.join(self.bkg_path, bin_filepath)

        with open(bin_filepath + '.txt', 'w') as f:
            f.write('\n'.join(lines))

        numpngw.write_png(bin_filepath + '.png', data_int)




alignment = Alignment()
alignment.ask_for_folders()
alignment.ask_for_bkg_image()
alignment.ask_best_or_all()
time.sleep(0.5)
if alignment.decision == 1:
    alignment.subtract_bkg_from_selected_folders()
    alignment.initial_ROIs()
    time.sleep(0.5)
    alignment.align_best()
else:
    alignment.single_ROI()
    alignment.align_everything()
