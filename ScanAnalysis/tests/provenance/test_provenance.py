"""Tests for the analysis provenance module."""

import json
from pathlib import Path


from scan_analysis.provenance.models import (
    AnalysisEntry,
    CodeVersion,
    ProvenanceFile,
    Software,
)
from scan_analysis.provenance.io import (
    get_provenance_path,
    log_provenance,
    read_provenance,
)
from scan_analysis.provenance.capture import (
    capture_code_version,
    capture_dependencies,
    get_current_user,
)


class TestModels:
    """Test Pydantic models for provenance."""

    def test_analysis_entry_minimal(self):
        """Test creating a minimal analysis entry."""
        entry = AnalysisEntry(columns_written=["col_a", "col_b"])
        assert entry.columns_written == ["col_a", "col_b"]
        assert entry.timestamp is not None
        assert entry.software is None
        assert entry.code_version is None

    def test_analysis_entry_full(self):
        """Test creating a full analysis entry."""
        entry = AnalysisEntry(
            columns_written=["centroid_x", "centroid_y"],
            software=Software(name="test_tool", version="1.0.0"),
            code_version=CodeVersion(
                repository="https://github.com/test/repo",
                commit="abc123",
                branch="main",
                dirty=False,
            ),
            dependencies={"numpy": "2.0.0", "pandas": "2.2.0"},
            config={"threshold": 0.5},
            notes="Test analysis",
            user="testuser",
        )
        assert entry.software.name == "test_tool"
        assert entry.code_version.commit == "abc123"
        assert entry.dependencies["numpy"] == "2.0.0"

    def test_provenance_file_append(self):
        """Test appending entries to provenance file."""
        prov = ProvenanceFile()
        assert len(prov.analyses) == 0

        entry1 = AnalysisEntry(columns_written=["col_a"])
        prov.append_entry(entry1)
        assert len(prov.analyses) == 1

        entry2 = AnalysisEntry(columns_written=["col_b"])
        prov.append_entry(entry2)
        assert len(prov.analyses) == 2

    def test_get_column_provenance(self):
        """Test getting provenance for a specific column."""
        prov = ProvenanceFile()
        entry1 = AnalysisEntry(
            columns_written=["col_a", "col_b"],
            notes="First analysis",
        )
        entry2 = AnalysisEntry(
            columns_written=["col_b", "col_c"],
            notes="Second analysis",
        )
        prov.append_entry(entry1)
        prov.append_entry(entry2)

        # col_b appears in both, should return most recent
        result = prov.get_column_provenance("col_b")
        assert result is not None
        assert result.notes == "Second analysis"

        # col_a only in first
        result = prov.get_column_provenance("col_a")
        assert result is not None
        assert result.notes == "First analysis"

        # col_d doesn't exist
        result = prov.get_column_provenance("col_d")
        assert result is None


class TestProvenancePath:
    """Test provenance file path generation."""

    def test_txt_file(self):
        """Test path for .txt file."""
        result = get_provenance_path("/data/scans/s123.txt")
        assert result == Path("/data/scans/s123.provenance.json")

    def test_csv_file(self):
        """Test path for .csv file."""
        result = get_provenance_path("/data/scans/data.csv")
        assert result == Path("/data/scans/data.provenance.json")

    def test_relative_path(self):
        """Test with relative path."""
        result = get_provenance_path("s456.txt")
        assert result == Path("s456.provenance.json")


class TestIO:
    """Test provenance file I/O operations."""

    def test_log_and_read_provenance(self, tmp_path):
        """Test logging and reading provenance."""
        data_file = tmp_path / "s123.txt"
        data_file.write_text("Shotnumber\tcol_a\n1\t10\n")

        # Log provenance
        success = log_provenance(
            data_file=data_file,
            columns_written=["col_a", "col_b"],
            software_name="test_tool",
            software_version="1.0.0",
            auto_capture_code=False,
            auto_capture_deps=False,
            auto_capture_user=False,
        )
        assert success

        # Verify file was created
        provenance_path = tmp_path / "s123.provenance.json"
        assert provenance_path.exists()

        # Read it back
        prov = read_provenance(data_file)
        assert prov is not None
        assert len(prov.analyses) == 1
        assert prov.analyses[0].columns_written == ["col_a", "col_b"]
        assert prov.analyses[0].software.name == "test_tool"

    def test_log_multiple_entries(self, tmp_path):
        """Test logging multiple provenance entries."""
        data_file = tmp_path / "s456.txt"
        data_file.write_text("Shotnumber\tcol_a\n1\t10\n")

        # First entry
        log_provenance(
            data_file=data_file,
            columns_written=["col_a"],
            software_name="tool1",
            auto_capture_code=False,
            auto_capture_deps=False,
            auto_capture_user=False,
        )

        # Second entry
        log_provenance(
            data_file=data_file,
            columns_written=["col_b"],
            software_name="tool2",
            auto_capture_code=False,
            auto_capture_deps=False,
            auto_capture_user=False,
        )

        prov = read_provenance(data_file)
        assert prov is not None
        assert len(prov.analyses) == 2
        assert prov.analyses[0].software.name == "tool1"
        assert prov.analyses[1].software.name == "tool2"

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading provenance when file doesn't exist."""
        data_file = tmp_path / "nonexistent.txt"
        result = read_provenance(data_file)
        assert result is None

    def test_empty_columns_rejected(self, tmp_path):
        """Test that empty columns_written is rejected."""
        data_file = tmp_path / "s789.txt"
        data_file.write_text("data")

        success = log_provenance(
            data_file=data_file,
            columns_written=[],
            auto_capture_code=False,
            auto_capture_deps=False,
            auto_capture_user=False,
        )
        assert not success


class TestExtractConfig:
    """Tests for the extract_config_from_analyzer function."""

    def test_extract_from_camera_config_pydantic(self):
        """Test extracting config from camera_config (Pydantic model)."""
        from scan_analysis.provenance import extract_config_from_analyzer
        from pydantic import BaseModel

        class MockConfig(BaseModel):
            roi: list = [0, 100, 0, 100]
            threshold: float = 0.5

        class MockAnalyzer:
            camera_config = MockConfig()

        analyzer = MockAnalyzer()
        config = extract_config_from_analyzer(analyzer)

        assert config is not None
        assert config["roi"] == [0, 100, 0, 100]
        assert config["threshold"] == 0.5

    def test_extract_from_generic_config(self):
        """Test extracting config from generic config attribute."""
        from scan_analysis.provenance import extract_config_from_analyzer

        class MockAnalyzer:
            config = {"setting1": "value1", "setting2": 42}

        analyzer = MockAnalyzer()
        config = extract_config_from_analyzer(analyzer)

        assert config is not None
        assert config["setting1"] == "value1"
        assert config["setting2"] == 42

    def test_extract_from_get_config_method(self):
        """Test extracting config from get_config() method."""
        from scan_analysis.provenance import extract_config_from_analyzer

        class MockAnalyzer:
            def get_config(self):
                return {"method_key": "method_value"}

        analyzer = MockAnalyzer()
        config = extract_config_from_analyzer(analyzer)

        assert config is not None
        assert config["method_key"] == "method_value"

    def test_returns_none_for_no_config(self):
        """Test returns None when no config is available."""
        from scan_analysis.provenance import extract_config_from_analyzer

        class MockAnalyzer:
            pass

        analyzer = MockAnalyzer()
        config = extract_config_from_analyzer(analyzer)

        assert config is None


class TestCapture:
    """Tests for automatic capture utilities."""

    def test_capture_code_version(self):
        """Test capturing git version (may be None if not in repo)."""
        # This test just checks it doesn't crash
        result = capture_code_version()
        # In GEECS-Plugins repo, this should return something
        if result is not None:
            assert isinstance(result, CodeVersion)
            # commit should be a hex string
            if result.commit:
                assert len(result.commit) >= 7

    def test_capture_dependencies(self):
        """Test capturing package versions."""
        deps = capture_dependencies(["pydantic", "nonexistent_package_xyz"])
        assert "pydantic" in deps
        assert "nonexistent_package_xyz" not in deps

    def test_get_current_user(self):
        """Test getting current user."""
        user = get_current_user()
        assert user is not None
        assert len(user) > 0


class TestJSONSchema:
    """Test that generated JSON matches the schema."""

    def test_minimal_example_structure(self, tmp_path):
        """Test that minimal log matches expected structure."""
        data_file = tmp_path / "test.txt"
        data_file.write_text("data")

        log_provenance(
            data_file=data_file,
            columns_written=["x", "y"],
            auto_capture_code=False,
            auto_capture_deps=False,
            auto_capture_user=False,
        )

        provenance_path = tmp_path / "test.provenance.json"
        content = json.loads(provenance_path.read_text())

        # Check schema structure
        assert "schema_version" in content
        assert content["schema_version"] == "0.1"
        assert "analyses" in content
        assert len(content["analyses"]) == 1
        assert "timestamp" in content["analyses"][0]
        assert "columns_written" in content["analyses"][0]
