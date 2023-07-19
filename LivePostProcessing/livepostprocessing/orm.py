# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:28:32 2023

@author: Reinier van Mourik
"""

from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from warnings import warn
from configparser import ConfigParser, NoSectionError
import re
import shutil

from typing import TYPE_CHECKING, Optional, Union
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
from .types import ShotNumber, DeviceName, ImageSubject, ImageFolderName

import pandas as pd
import numpy as np

from .utils import (find_device_image_folders, read_scalar_data, parse_run_date, 
                    find_undulator_folder, read_imaq_image
                   ) 

undulator_data_folder = find_undulator_folder()


class Scan:
    """ Represents an undulator scan
    
    """

    def __init__(self, run_date: Union[str, date], scan: int, create_image_directory=True):
        """
        Parameters
        ----------
        run_date : str|date
            A run date either as a date or datetime object, or a string 
            representing the date in a parseable format or as '23_0412' format
            used in the undulator data.
        scan : int
            the scan number
            
        """
        
        # date of this run.
        # get the 'yy_mmdd' form with the self.run_id property
        self.run_date: date
        if isinstance(run_date, date):  # matches date or datetime object
            self.run_date = date(run_date.year, run_date.month, run_date.day)
        else:
            self.run_date = parse_run_date(run_date)

        # scan number
        self.number: int = scan

        # scalar data
        self.scalar_data = pd.DataFrame([], 
                                        index=pd.MultiIndex.from_tuples([], names=['run', 'scan', 'shot']),
                                        columns=pd.MultiIndex.from_tuples([], names=['device', 'metric'])
                                       ) 
        try:
            self.load_scalar_data()
        except Exception as err:
            warn(f"Could not load ScanData txt file: {err}")
        
        # scan info, including scan parameter and its steps
        self.scan_info: dict[str, str] = {}
        try:
            self.load_scan_info()
        except Exception as err:
            warn(f"Could not load ScanInfo ini file: {err}")

        # device settings for this scan
        self.device_config: dict[str, dict[str, str]] = {}
        try:
            self.load_device_config()
        except Exception as err:
            warn(f"Could not load device config file: {err}")

        # catalog all images in this scan
        self.image_directory = pd.Series([], dtype=object, index=pd.MultiIndex.from_tuples([], names=('shot', 'device', 'subject')))
        if create_image_directory:
            self.create_image_directory()

        # list of shots in this scan.
        self.shots: dict[ShotNumber, Shot] = {}
        for _, _, shotnumber in self.scalar_data.index:
            self.shots[shotnumber] = Shot(self, shotnumber)

    def __repr__(self):
        return f"Scan('{self.run_id}', {self.number:d})"

    def load_scalar_data(self):
        self.scalar_data = read_scalar_data(self.path/f"ScanDataScan{self.number:03d}.txt", self.run_id)

    def load_scan_info(self):
        config_parser = ConfigParser()
        config_parser.read(self.path/f"ScanInfoScan{self.number:03d}.ini")

        try:
            self.scan_info.update({key: value.strip("'\"")
                                   for key, value in config_parser.items("Scan Info")})
        except NoSectionError:
            warn("ScanInfo file does not have a 'Scan Info' section.")

    def load_device_config(self):
        """ Loads device configurations from ECS Live Dumps file.
        """
        config_parser = ConfigParser()
        config_parser.read(self.path.parent.parent/'ECS Live Dumps'/f"Scan{self.number:d}.txt")
        for section_name, section in config_parser.items():
            if "Device Name" not in section:
                continue
            
            device_name = section["Device Name"].strip('"')
            self.device_config[device_name] = {}
            for option, value in section.items():
                self.device_config[device_name][option] = value.strip('"')

    @property
    def run_id(self) -> str:
        """ run date in yy_mmdd format """
        return self.run_date.strftime('%y_%m%d')

    @property
    def path(self) -> Path:
        return (undulator_data_folder/f"Y{self.run_date:%Y}"/self.run_date.strftime('%m-%b')/
                self.run_id/'scans'/f"Scan{self.number:03d}"
               )

    @property
    def analysis_path(self) -> Path:
        """ Path to the analysis/ScanXXX folder that contains additional images """
        return self.path.parents[1] / 'analysis' / f"Scan{self.number:03d}"

    @property
    def s_file_path(self) -> Path:
        """ Path to the sXX.txt file that contains scalar data and custom metrics """
        return self.path.parents[1] / 'analysis' / f"s{self.number:d}.txt"

    def scalar_data_groupedby_scan_parameter_step(self) -> DataFrameGroupBy:
        """ Helper function to group scalar data DataFrame by scan parameter step.
        
        For example, to get averages of all columns by scan parameter step:
            averages = scan.scalar_data_groupedby_scan_parameter_step().agg('mean')
        which returns a DataFrame with as index the value of the (nominal) scan parameter

        """
        # an array of a number identfying the scan parameter step group
        # 0, 0, 0 ... 0, 1, 1, ... 1, 2, ... 2, ... 
        scan_parameter_step_numbers = (self.scalar_data.index.get_level_values('shot') - 1) // int(self.scan_info['shots per step'])
        # the parameter set point for each shot
        scan_parameter_nominal_values = float(self.scan_info['start']) + float(self.scan_info['step size']) * scan_parameter_step_numbers

        return self.scalar_data.groupby(scan_parameter_nominal_values)

    def create_image_directory(self) -> None:
        """ Scan through image folders and index them.

        Saves a pd.Series to self.image_directory, with index ('shot', 'device', 'subject')
        
        """
        image_filename_regex = re.compile(f"Scan{self.number:03d}_"    # match ScanXXX exactly
                                          r"(?P<device_subject>.*)_"   # usually matches image folder name, but not always, like Scan011_U_HiResMagCam-interp_002.txt for subject interpDiv
                                          r"(?P<shot_number>\d{3,})"   # if shot number > 999, it can be more than 3 digits
                                          r"(?P<suffix>_.*)?"          # there can be an additional word after the shot number, like in Scan011_U_GhostWFS_002_wavefront.png
                                          r"\.(?P<format>\w+)"         # png, tif, txt, dat, etc.
                                         )

        image_directory_entries: dict[tuple[ShotNumber, DeviceName, ImageSubject], Path] = {}
        accessed_subfolders: set[ImageFolderName] = set()
        # search by going through device names in the scalar data instead of going through folders, because
        # some image folder names are DeviceName-Subject, and some `DeviceName`s have hyphens so it's harder
        # to separate Devicename and Subject.
        for device in self.scalar_data.columns.get_level_values('device').unique():
            subjects: dict[ImageSubject, ImageFolderName] = find_device_image_folders(self.path, device)
            if len(subjects) > 0:
                for subject, image_folder_name in subjects.items():
                    # image_folder is either DeviceName, if subject is None, '', or 'raw', or 
                    # DeviceName-Subject , for example for U_HiResMagCam-interpSpec
                    if not (self.path/image_folder_name).exists():
                        continue

                    accessed_subfolders.add(image_folder_name)

                    for filepath in (self.path/image_folder_name).iterdir():
                        # ignore certain files.
                        if filepath.name in ['Thumbs.db']:
                            continue

                        m = image_filename_regex.match(filepath.name)
                        if m is not None:
                            # make sure there isn't more than one file related to this shot/device/subject.
                            key = (ShotNumber(int(m['shot_number'])), device, subject)
                            assert key not in image_directory_entries
                            image_directory_entries[key] = filepath
                        else:
                            warn(f"Unexpected filename: {filepath.name}")

        if image_directory_entries:
            # pd.Series of datatype Path
            self.image_directory = pd.Series(image_directory_entries).rename_axis(('shot', 'device', 'subject'))

            # check whether all subfolders of scan folder were accessed
            self.not_accessed_subfolders = {ImageFolderName(p.name) 
                                            for p in self.path.iterdir() 
                                            if p.is_dir()
                                            and p.name not in accessed_subfolders
                                        }
            if self.not_accessed_subfolders: 
                warn("The following subfolders were not accessed: " + ', '.join(self.not_accessed_subfolders))


class Shot:
    def __init__(self, parent_scan: Scan, shot_number: ShotNumber):
        self.parent: Scan = parent_scan
        self.number: ShotNumber = shot_number

        self.images: dict[DeviceName, dict[ImageSubject, Image]] = {}
        self.create_image_directory()

    def __repr__(self):
        return f"Shot({self.parent}, {self.number:d})"

    @property
    def scalar_data(self) -> pd.Series:
        return self.parent.scalar_data.loc[(self.parent.run_id, self.parent.number, self.number)]

    @property
    def datetime(self) -> Optional[datetime]:
        timestamp = self.scalar_data.get(('DateTime', 'Timestamp'))
        if timestamp is not None:
            return (datetime(1904, 1, 1, 0, 0, 0, 0, ZoneInfo('UTC')) + timedelta(seconds=timestamp)).astimezone(ZoneInfo('US/Pacific'))
        else:
            return None

    def create_image_directory(self) -> None:
        for (device, subject), image_filepath in self.parent.image_directory.get(self.number, pd.Series([], dtype=object)).items():
            if device not in self.images:
                self.images[device] = {}
            self.images[device][subject] = Image(self, device, subject, image_filepath) 

class Image:
    """ Represents an image captured by a device during a Scan. 

    Does not contain the image payload itself but includes a method to load it.
    
    """
    def __init__(self, parent_shot: Shot, device: DeviceName, subject: Optional[ImageSubject] = None, path: Optional[Path] = None):
        self.parent: Shot = parent_shot
        self.device: str = device
        self.subject: str = subject if subject is not None else 'raw'

        self.path: Path
        if path is None:
            self.path = self._find_image_path(self.parent, self.device, self.subject)
        else:
            self.path = path
        
    @property
    def format(self) -> str:
        return self.path.suffix.lstrip('.').lower()

    @property
    def parent_scan(self) -> Scan:
        return self.parent.parent

    @property
    def image_folder_name(self) -> str:
        if not self.subject or self.subject == 'raw':
            return self.device
        else:
            return f"{self.device}-{self.subject}"

    def load(self) -> np.ndarray:
        if self.format.lower() == 'npy':
            return np.load(self.path)
        else:
            return read_imaq_image(self.path)

    @staticmethod
    def _find_image_path(shot: Shot, device: str, subject: Optional[str] = None) -> Path:
        """ Finds images corresponding to a particular device and subject for this shot.
        """
        
        # image folder name is just DeviceName for the 'raw' subject, or 
        # DeviceName-Subject otherwise. 
        image_folder_name: str
        if not subject or subject == 'raw':
            image_folder_name = device
        else:
            image_folder_name = f"{device}-{subject}"

        image_folder: Path = shot.parent.path/image_folder_name
        # look for files with the right name, and any extension.
        image_filepaths: list[Path] = list(
            image_folder.glob(f"Scan{shot.parent.number:03d}_{device}*_{shot.number:03d}*")
        )
        if len(image_filepaths) < 1:
            raise ValueError(f"No images found for Scan{shot.parent.number:03d}/Shot{shot.number:03d}/{image_folder_name}")
        elif len(image_filepaths) > 1:
            raise ValueError(f"More than one image found for Scan{shot.parent.number:03d}/Shot{shot.number:03d}/{image_folder_name}")
        else:
            # grab first and only image_filepath
            return image_filepaths[0]



class SFile:
    def __init__(self, scan: Scan):
        self.scan = scan
        self.load_s_file()

    @property
    def path(self):
        return self.scan.path.parent.parent/'analysis'/f"s{self.scan.number:d}.txt"
    
    def load_s_file(self):
        self.scalar_data = read_scalar_data(self.path, run_id=self.scan.run_id, unable_to_split_column_name='metric')

    def save_s_file(self):
        # first back up the previous one
        try:
            backup_folder = self.path.parent/'s_file_backup'
            backup_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.path, (backup_folder/self.path.name).with_suffix(f".bk{datetime.now():%y%m%d-%H%M}.txt"))
        except Exception as err:
            raise IOError(f"Unable to create backup for {self.path}: {err}.")

        # save scalar data to s file path
        df = self.scalar_data.copy()
        # condense the MultiIndex column header to "{device} {variable}" format
        df.columns = [f"{c[0]} {c[1]}".strip(' ') for c in df.columns]
        # remove run_id, and rename shot back to Shotnumber
        df.index = df.index.droplevel('run').rename(['scan', 'Shotnumber'])
        # save to csv
        df.to_csv(self.path, sep='\t')

