import pytest
from pathlib import Path
import numpy as np
from image_analysis.offline_analyzers.Undulator.VisaEBeam import VisaEBeam

@pytest.mark.parametrize("camera_name", [
    "UC_VisaEBeam1",
    "UC_VisaEBeam2",
    "UC_VisaEBeam3",
    "UC_VisaEBeam4",
    "UC_VisaEBeam5",
    "UC_ALineEBeam3"
])
def test_analyze_image_file_visaebeam(camera_name):
    # Resolve test image path relative to this test file
    current_dir = Path(__file__).resolve().parent.parent
    image_name = f"{camera_name}_001.png"
    test_img_path = current_dir / "data" / "VisaEBeam_test_data" / image_name

    assert test_img_path.exists(), f"Test image not found: {test_img_path}"

    analyzer = VisaEBeam(camera_name=camera_name)
    analyzer.use_interactive = True
    result = analyzer.analyze_image_file(image_filepath=test_img_path)

    # Validate result structure
    assert isinstance(result, dict)
    assert "processed_image" in result
    assert "analyzer_input_parameters" in result

    image = result["processed_image"]
    config_inputs = result["analyzer_input_parameters"]

    assert isinstance(image, np.ndarray)
    assert image.ndim == 2

if __name__ == "__main__":
    pytest.main()