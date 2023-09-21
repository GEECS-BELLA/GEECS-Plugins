from __future__ import annotations

from pathlib import Path
import re
from collections import namedtuple
from datetime import date, datetime
from dateutil.parser import parse as date_parse
from dateutil.parser import ParserError as DateParserError
from configparser import ConfigParser, NoSectionError
from typing import Optional, TYPE_CHECKING, Union
import png
from imageio.v3 import imread

if TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray
from .types import ImageSubject, ImageFolderName, DeviceName

import pandas as pd
import numpy as np

# from dotenv import dotenv_values

def find_undulator_folder() -> Path:
    undulator_data_folder = Path("Z:/data/Undulator")
    if not undulator_data_folder.exists():
        undulator_data_folder = Path(r"C:\Users\ReiniervanMourik\Documents\tau_systems\data\Undulator")
    if not undulator_data_folder.exists():
        raise FileNotFoundError("Can't find undulator data folder.")

    return undulator_data_folder


def collect_scan_info() -> None:
        
    config_parser = ConfigParser()
    
    all_scan_info: list[dict] = []
    
    for scan_folder_path in iterate_scan_folders(find_undulator_folder()):
        
        scan_info_path = scan_folder_path / f"ScanInfo{scan_folder_path.name}.ini"

        config_parser.read(scan_info_path)
        
        scan_info = {'date_folder': scan_info_path.parts[-4]}
        
        try:
            scan_info.update({key: value.strip("'\"") 
                              for key, value in config_parser.items("Scan Info")
                             }
                            )
        except NoSectionError as err:
            print(f"Warning: {scan_info_path} does not have a 'Scan Info' section.")
    
        except Exception as err:
            print(f"Error parsing {scan_info_path}: {err}")
    
        all_scan_info.append(scan_info)
    
    pd.DataFrame(all_scan_info).set_index(['date_folder', 'scan no']).to_csv(find_undulator_folder() / "ScanInfo.csv", sep='\t')


def find_device_image_folders(scan_folder: Path, device: DeviceName) -> dict[ImageSubject, ImageFolderName]:
    """ find device image folders.
        folder can be simply the device name, e.g. 'UC_DiagnosticsPhosphor'
        or folder can be "<devicename>-<subject>", e.g. 'UC_BCaveMagSpecCam3-interp'
        or both.
        (I do this here instead of parsing image_folders, because some 
        devices have hyphens in them)
        
        Returns
        -------
        device_image_folders : dict[str, str]
            dictionary of subject: image_folder_name
    """

    device_image_folders: dict[ImageSubject, ImageFolderName] = {}
    device_image_regex = re.compile(f"^{device}(?:-(?P<subject>.*))?$")
    for image_folder in scan_folder.glob(f"{device}*"):
        image_folder_name = ImageFolderName(image_folder.name)
        m = device_image_regex.match(image_folder_name)
        if m is not None:
            if m['subject'] is None:
                device_image_folders[ImageSubject('raw')] = image_folder_name
            else:
                device_image_folders[ImageSubject(m['subject'])] = image_folder_name
    
    return device_image_folders


column_name_regex = re.compile(
    r"^(?P<device>[\w\-]+) "
    r"(?P<metric>[\w \.#\-\(\)]+)"
    r"(?: Alias:(?P<alias>.+))?$"
)


def deconstruct_scandata_column_name(column_name: str) -> dict:
    """ Extract device and metric from the column name in a ScanData file.

    Returns
    -------
    dict 
        with keys 'device', 'metric', and 'alias'.

    """

    m = column_name_regex.match(column_name)
    
    if m is None:
        raise ValueError(f"Could not deconstruct column name '{column_name}'")

    return {'device': m.group('device'),
            'metric': m.group('metric'),
            'alias': m.group('alias'),
           }


def read_scalar_data(scalar_data_path: Union[str, PathLike], 
                     run_id: Optional[str] = None, 
                     unable_to_split_column_name: str = 'raise'
                    ) -> pd.DataFrame:
    """ Returns a DataFrame with columns a MultiIndex of levels 'device' and 'metric'
        and index a MultiIndex with levels 'run', 'scan', 'shot'
    """
    if run_id is None:
        run_id, scans_literal, scan_folder_name, scan_file_name = Path(scalar_data_path).parts[-4:]
        assert scans_literal == 'scans'
        assert re.match(r'Scan(\d+)', scan_folder_name) is not None
        
    def split_column_name_into_device_and_metric(column_name: str):
        try:
            parts = deconstruct_scandata_column_name(column_name)
            return (parts['device'], parts['metric'])
        except ValueError as err:
            if unable_to_split_column_name == 'raise':
                raise err
            elif unable_to_split_column_name == 'metric':
                return ('', column_name)
            elif unable_to_split_column_name == 'device':
                return (column_name, '')
            else:
                raise ValueError(f"Unknown value for 'error' during ValueError({err})")

    df = pd.read_csv(scalar_data_path, sep='\t', )
    if not len(df):
        return pd.DataFrame([], index=pd.MultiIndex.from_tuples([], names=('run', 'scan', 'shot')),
                                columns=pd.MultiIndex.from_tuples([], names=('device', 'metric')),
                           )

    df = (df.assign(run_id=run_id)
            .assign(Shotnumber=df.Shotnumber.astype(int),
                    scan=df.scan.astype(int)
                   )
            .set_index(['run_id', 'scan', 'Shotnumber'])
            .rename_axis(index=['run', 'scan', 'shot'])
            .rename(columns=split_column_name_into_device_and_metric)
            )

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df.rename_axis(columns=['device', 'metric'])


def get_run_folder(run: Union[str, date]) -> Path:
    """ Return path to folder for the run, i.e. date folder.

    Parameters
    ----------
    run : str|date
        The date of the scan, i.e. name of the run folder, either as a date, 
        string that can be parsed into a date, or a string in 'yy_mmdd' format.

    """

    run_date: date
    if isinstance(run, date):  # matches date or datetime object
        run_date = date(run.year, run.month, run.day)
    else:
        run_date = parse_run_date(run)

    return (find_undulator_folder()/f"Y{run_date:%Y}"/run_date.strftime('%m-%b')/
            run_date.strftime('%y_%m%d')
           )

def get_scan_folder(run: Union[str, date], scan: int) -> Path:
    """ Return path to folder for the scan with given run_date and scan number.

    Parameters
    ----------
    run : str|date
        The date of the scan, i.e. name of the run folder, either as a date, 
        string that can be parsed into a date, or a string in 'yy_mmdd' format.
    scan : int
        scan number

    """

    return get_run_folder(run)/'scans'/f"Scan{scan:03d}"



def parse_run_date(run_date: str):
    """ Parses a date string of format 'yy_mmdd' or any other known format.
    """
    
    try:
        # see if date is in a commonly understood format
        return date_parse(run_date).date()
    except DateParserError:
        try: 
            # otherwise see if it's in 23_0412 format
            return datetime.strptime(run_date, '%y_%m%d').date()
        except ValueError:
            raise ValueError(f"Could not parse run_date '{run_date}'")


def iterate_scan_folders(start_folder_path: Union[str, PathLike], recursion_level: Optional[str] = None):
    r""" Searches the Undulator data folder for ScanXXX folders. 

    Specifically, those that look like
        Undulator\YXXXX\XX-Xxx\XX_XXXX\scans\ScanXXX
    
    Parameters
    ----------
    start_folder_path : str or Path
        the Undulator folder or a subfolder thereof.
    recursion_level : str
        If this is a subfolder of Undulator (such as a specific month folder),
        recursion_level should be 'year', 'month', 'date', 'scans', or 'scan'

    Generates
    ---------
    scan_folder_path : Path

    """
    start_folder_path = Path(start_folder_path)

    LevelInfo = namedtuple('LevelInfo', ['folder_regex', 'next_level'])
    level_info = {'Undulator':  LevelInfo(re.compile(r"^Undulator$"), 'year'),
                  'year':       LevelInfo(re.compile(r"^Y\d{4}$"), 'month'),
                  'month':      LevelInfo(re.compile(r"^\d{2}-\w{3}$"), 'date'),
                  'date':       LevelInfo(re.compile(r"^\d{2}_\d{4}$"), 'scans'),
                  'scans':      LevelInfo(re.compile(r"scans"), 'scan'),
                  'scan':       LevelInfo(re.compile(r"^Scan(?P<scan>\d{3})$"), None),
                 }

    # detect recursion level if not given
    if recursion_level is None:
        for recursion_level, (folder_regex, _) in level_info.items():
            if folder_regex.match(start_folder_path.name):
                break
        else:
            raise ValueError("Unable to determine recursion level.")

    # meat of the recursion
    if recursion_level == 'scan':
        # we've reached the scanfolder
        yield start_folder_path        

    else:
        # scan one level deeper
        subfolder_level = level_info[recursion_level].next_level
        for subfolder_path in Path(start_folder_path).iterdir():
            # ignore any files or folders that don't match the subfolder regex
            if not level_info[subfolder_level].folder_regex.match(subfolder_path.name):
                continue

            # search this subfolder recursively for scan folders
            yield from iterate_scan_folders(subfolder_path, recursion_level=subfolder_level)



class NotAPath(Path().__class__):
    """ A Path instance that evaluates to false in, for example, if statements.
    """
    def __bool__(self):
        return False
