import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from geecs_python_api.controls.devices.geecs_device import api_error
import geecs_python_api.tools.images.ni_vision as ni


def average_images(images_folder: Union[Path, str], n_images: int = 0, file_extension: str = '.png') \
        -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not images_folder:
        return None, None

    images_folder = Path(images_folder)
    images = list_files(images_folder, n_images, file_extension)

    # run averaging
    if images:
        try:
            with ProgressBar(max_value=len(images)) as pb:
                avg_image = ni.read_imaq_image(images[0])
                data_type: str = avg_image.dtype.name
                avg_image = avg_image.astype('float64')
                pb.increment()

                if len(images) > 1:
                    for it, image_path in enumerate(images[1:]):
                        image_data = ni.read_imaq_image(image_path)
                        image_data = image_data.astype('float64')
                        alpha = 1.0 / (it + 2)
                        beta = 1.0 - alpha
                        avg_image = cv2.addWeighted(image_data, alpha, avg_image, beta, 0.0)
                        pb.increment()

            return avg_image, avg_image.round().astype(data_type)

        except Exception as ex:
            api_error.error(str(ex), 'Failed to calculate average image')
            return None, None


def list_files(folder: Union[Path, str], n_files: Optional[int] = None, file_extension: str = '.png') -> list[Path]:
    # file extension
    if not file_extension:
        return None

    folder = Path(folder)
    file_extension = file_extension.lower().strip()
    if file_extension.startswith('.'):
        file_extension = file_extension[1:]

    # if file_extension not in ['png', 'jpg', 'jpeg', 'tiff', 'tif']:
    #     return None

    # list files
    files = sorted(glob.glob(str(folder / f'*.{file_extension}')), key=lambda f: Path(f).parts[-1])

    if n_files > 0:
        files = files[-n_files:]

    return [Path(p) for p in files]


if __name__ == '__main__':
    _folder = r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\05-May\23_0509\scans\Scan022\UC_VisaEBeam1'

    image, _ = average_images(_folder)
    plt.figure()
    plt.imshow(image)
    plt.show(block=True)
