""" Example scanner tools integration test with pytest

pytest is a testing framework that is very popular due to its simplicity.

"""

import pytest
import numpy as np

from pathlib import Path

from geecs_python_api.analysis.scans.scan_data import ScanData, ScanTag
from image_analysis.analyzers.HASO_himg_has_processor import HASOHimgHasProcessor, FilterParameters
import image_analysis.third_party_sdks.wavekit_43.wavekit_py as wkpy

def get_path_to_haso_file():
    st = ScanTag(2025, 2, 19, 3, experiment='Undulator')
    s_data = ScanData(tag=st)
    path_to_file = s_data.get_device_shot_path(tag=st, device_name='U_HasoLift', shot_number=10, file_extension = 'himg')
    return path_to_file

path_to_example_has_file = Path('Z:/data/Undulator/Y2025/02-Feb/25_0219/scans/Scan002/U_HasoLift/Scan002_U_HasoLift_019_raw.has')
path_to_example_himg_file = get_path_to_haso_file()
path_to_bkg_has_file = Path('Z:/data/Undulator/Y2025/02-Feb/25_0219/analysis/Scan002/U_HasoLift/HasoAnalysis/Scan002_U_HasoLift_017_raw.has')

def test_get_path_to_haso_file():
    path_to_file = get_path_to_haso_file()
    return path_to_file.is_file()

def test_process_haso_class_generation():
    haso_processor = HASOHimgHasProcessor()
    return True

def test_generate_slopes_from_himg():
    haso_processor = HASOHimgHasProcessor()
    print(path_to_example_himg_file)
    slopes_object = haso_processor.create_slopes_object_from_himg(image_file_path = path_to_example_himg_file)
    assert isinstance(slopes_object, wkpy.HasoSlopes)

def test_load_slopes_from_has():
    haso_processor = HASOHimgHasProcessor()
    slopes_object = haso_processor.load_slopes_from_has_file(path_to_example_has_file)
    assert isinstance(slopes_object, wkpy.HasoSlopes)

def analyze_image(file_path):
    haso_processor = HASOHimgHasProcessor()
    haso_processor.analyze_image(file_path)
    return haso_processor.raw_slopes

def test_analyze_image_has():
    assert isinstance(analyze_image(path_to_example_has_file), wkpy.HasoSlopes)

def test_analyze_image_himg():
    assert isinstance(analyze_image(path_to_example_himg_file), wkpy.HasoSlopes)

def analyze_image_with_postprocessing(file_path):
    haso_processor = HASOHimgHasProcessor(background_path = path_to_bkg_has_file)
    # haso_processor = HASOHimgHasProcessor()
    haso_processor.filter_params = FilterParameters(apply_tiltx_filter=True, apply_tilty_filter=True, apply_curv_filter=True)
    haso_processor.analyze_image(file_path)
    return haso_processor.processed_slopes

def test_analyze_image_with_postprocessing():
    assert isinstance(analyze_image(path_to_example_has_file), wkpy.HasoSlopes)

def test_analyze_image_returns_correct_types():
    haso_processor = HASOHimgHasProcessor(background_path = path_to_bkg_has_file)
    haso_processor.filter_params = FilterParameters(apply_tiltx_filter=True, apply_tilty_filter=True, apply_curv_filter=True)
    result = haso_processor.analyze_image(path_to_example_has_file)

    # Unpack the returned tuple.
    raw_slopes, processed_slopes, raw_phase, processed_phase, intensity = result

    # Assert that raw_slopes and processed_slopes are instances of wkpy.HasoSlopes.
    assert isinstance(raw_slopes, wkpy.HasoSlopes), "raw_slopes is not a HasoSlopes object"
    assert isinstance(processed_slopes, wkpy.HasoSlopes), "processed_slopes is not a HasoSlopes object"

    # Assert that raw_phase, processed_phase, and intensity are numpy arrays.
    assert isinstance(raw_phase, np.ndarray), "raw_phase is not an np.ndarray"
    assert isinstance(processed_phase, np.ndarray), "processed_phase is not an np.ndarray"
    assert isinstance(intensity, np.ndarray), "intensity is not an np.ndarray"

if __name__ == "__main__":
    pytest.main()