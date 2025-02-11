# ###
# This is scratch document to sketch out one path towards implementing an XOPT optimization
# within the geecs python frame work. The logic is listed below:
#
# The intention is to start a 'noscan' then intermittently pause the scan. During this time,
# an optimization function will evaluate the 'cost function'. Then, xopt will take this information,
# calculate a next step, and then commands will be sent to the optimization parameters. Upon completion,
# the scan will resume. The code below is a sketch of a way to do this (without the xopt calculation implemented).
#
# One idiosyncrocy of the geecs python data aqcuition is that there is no "shot number" getting incremented.
# So, if images or other non scalar data are being acquired as part of the optimization, some care
# is needed to make sure the generated data is tracked properly
# ###




# Example usage for DataLogger
experiment_dir = 'Undulator'
shot_control_device = 'U_DG645_ShotControl'
scan_manager = ScanManager(experiment_dir=experiment_dir, shot_control_device = shot_control_device)

file1 = 'HiResOnly.yaml'

scan_manager.reinitialize(file1)

#test scan for a composite variable scan below, combined with using the reinitialize method
scan_config = {'device_var': 'noscan', 'start': -1, 'end': 1, 'step': 2, 'wait_time': 105.5, 'additional_description':'Testing out new python data acquisition module'}
# scan_manager.start_scan_thread(scan_config=scan_config)

# Start the scan thread
scan_manager.start_scan_thread(scan_config=scan_config)
scan_manager.pause_scan()


# Initialize shot tracking variables
dev_name = 'UC_HiResMagCam'
data_path = scan_manager.data_interface.local_scan_dir_base / scan_manager.data_interface.next_scan_folder

def get_new_files(data_path, last_shot_number):
    """Retrieve new files based on shot numbers."""
    new_files = []
    for file in data_path.glob(f"{dev_name}*"):  # Modify pattern as needed
        # Convert file to string and extract shot number
        shot_number = extract_shot_number(str(file))  # Adjust splitting logic if needed
        if shot_number > last_shot_number:
            new_files.append((shot_number, file))
    return sorted(new_files, key=lambda x: x[0])  # Sort by shot number for orderly processing

def avg_images(files):
    """
    Load all images and average them.

    Args:
        list of file paths of Imaq png images

    Returns:
        average of the images in uint16.
    """
    images = []

    for file in files:
        # print(f'file: {file}')
        image = read_imaq_png_image(file[1])
        images.append(image)

    return np.mean(images, axis=0).astype(np.uint16)  # Keep 16-bit format for the averaged image
 
collection_time = 5
last_shot_number = 0  # Keep track of the last shot number

# Loop to handle scan cycles and file processing
for i in range(10):
    # Resume scan
    scan_manager.resume_scan()
    time.sleep(collection_time)
    scan_manager.pause_scan()

    # Retrieve and process new files
    new_files = get_new_files(data_path / dev_name, last_shot_number)
    avg_image = avg_images(new_files)

    res = analyze_labview_image(dev_name, avg_image, None)
    print(f'res: {res[1]}')
    # Update last shot number for the next iteration
    if new_files:
        last_shot_number = new_files[-1][0]  # Last shot number processed
    
    scan_manager.data_logger.bin_num += 1

scan_manager.resume_scan()
scan_manager.stop_scanning_thread()