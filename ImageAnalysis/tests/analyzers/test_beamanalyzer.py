import pytest
from pathlib import Path
import numpy as np
from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
from image_analysis.config_loader import set_config_base_dir


@pytest.mark.parametrize(
    "camera_name",
    [
        "UC_VisaEBeam1",
        "UC_VisaEBeam2",
        "UC_VisaEBeam3",
        "UC_VisaEBeam4",
        "UC_VisaEBeam5",
        "UC_ALineEBeam3",
    ],
)
def test_analyze_image_file_visaebeam(camera_name):
    """Test BeamAnalzyer for HTU e-beam images."""
    # Resolve test image path relative to this test file
    current_dir = Path(__file__).resolve().parent.parent

    geecs_plugins_dir = current_dir.parent.parent
    set_config_base_dir(geecs_plugins_dir / "image_analysis_configs")

    image_name = f"{camera_name}_001.png"
    test_img_path = current_dir / "data" / "VisaEBeam_test_data" / image_name

    assert test_img_path.exists(), f"Test image not found: {test_img_path}"

    image_analyzer = BeamAnalyzer(camera_config_name=camera_name)
    image_analyzer.use_interactive = True
    result = image_analyzer.analyze_image_file(image_filepath=test_img_path)

    # Validate result structure
    assert isinstance(result, dict)
    assert "processed_image" in result
    assert "analyzer_input_parameters" in result

    image = result["processed_image"]

    assert isinstance(image, np.ndarray)
    assert image.ndim == 2


if __name__ == "__main__":
    import sys

    # Run only this specific test file
    pytest.main([sys.argv[0]])
