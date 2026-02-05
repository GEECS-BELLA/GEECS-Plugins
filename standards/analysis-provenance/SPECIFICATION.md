# Analysis Provenance Standard.

**Version:** 0.1 (Draft)
**Status:** Experimental
**Last Updated:** 2026-02-04

## 1. Overview

This standard defines a format for tracking the provenance of derived data columns in analysis files. It enables reproducibility by recording what software, configuration, and code version produced each derived value.

### 1.1 Goals

- **Reproducibility**: Enable recreation of analysis results years later
- **Traceability**: Track which tools modified which data columns
- **Simplicity**: Low barrier to adoption across different languages and tools
- **Extensibility**: Support minimal logging while allowing rich metadata

### 1.2 Non-Goals

- Replacing version control for code
- Tracking raw data acquisition (handled by separate systems)
- Enforcing specific analysis workflows

## 2. File Format

### 2.1 Location

Provenance files MUST be co-located with their data file using the same base name:

| Data File | Provenance File |
|-----------|-----------------|
| `s123.txt` | `s123.provenance.json` OR `s123.provenance.yaml` |
| `analysis.csv` | `analysis.provenance.json` OR `analysis.provenance.yaml` |

### 2.2 Format

Both JSON and YAML are valid formats.

- Implementations MUST support reading JSON (`.provenance.json`)
- Implementations SHOULD support reading YAML (`.provenance.yaml`)
- Implementations MAY write in either format

When both formats exist for the same data file, JSON takes precedence.

### 2.3 Encoding

Files MUST be encoded in UTF-8.

## 3. Schema

### 3.1 Root Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | YES | Version of this specification (e.g., `"0.1"`) |
| `analyses` | array | YES | List of analysis entries, ordered chronologically |

### 3.2 Analysis Entry

Each entry in the `analyses` array represents one analysis operation that wrote to the data file.

#### Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | string | YES | ISO 8601 datetime when analysis was performed |
| `columns_written` | array[string] | YES | Column names that were added or modified |

#### Recommended Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `software` | object | NO | Software identification (see §3.3) |
| `code_version` | object | NO | Source code identification (see §3.4) |

#### Optional Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dependencies` | object | NO | Key package versions as `{name: version}` |
| `config` | object | NO | Configuration used for analysis |
| `config_ref` | string | NO | Path to external configuration file |
| `notes` | string | NO | Human-readable context or comments |
| `user` | string | NO | Username or identifier of who ran the analysis |

### 3.3 Software Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | YES | Name of the software, tool, or script |
| `version` | string | NO | Version string (e.g., `"1.2.3"`, `"unknown"`) |

### 3.4 Code Version Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repository` | string | NO | Repository URL (e.g., GitHub URL) |
| `commit` | string | NO | Git commit SHA (full or abbreviated) |
| `branch` | string | NO | Branch name |
| `dirty` | boolean | NO | `true` if there were uncommitted changes |

## 4. Behavior

### 4.1 Append-Only

The `analyses` array is append-only. New analysis entries MUST be appended to the end of the array. Existing entries MUST NOT be modified or deleted.

### 4.2 Current State

When the same column appears in multiple entries, the **most recent entry** (last in the array) represents the authoritative provenance for the current data values.

### 4.3 Overwrites

When an analysis modifies existing columns (rather than adding new ones), the new entry simply lists those columns in `columns_written`. The history of previous writes is preserved in earlier entries.

### 4.4 Concurrent Access

Implementations SHOULD use appropriate file locking mechanisms when writing to prevent data corruption from concurrent writes.

### 4.5 Missing Provenance

Columns in the data file that do not appear in any provenance entry are considered to have "unknown provenance" (e.g., columns from raw data acquisition or legacy analyses).

## 5. Examples

### 5.1 Minimal Example

The simplest valid provenance file:

```json
{
  "schema_version": "0.1",
  "analyses": [
    {
      "timestamp": "2026-02-04T20:30:00Z",
      "columns_written": ["centroid_x", "centroid_y"]
    }
  ]
}
```

### 5.2 Full Example

A provenance file with all recommended fields:

```json
{
  "schema_version": "0.1",
  "analyses": [
    {
      "timestamp": "2026-02-04T14:30:00Z",
      "columns_written": [
        "UC_HiResMagCam peak_energy",
        "UC_HiResMagCam charge"
      ],
      "software": {
        "name": "scan_analysis",
        "version": "0.2.0"
      },
      "code_version": {
        "repository": "https://github.com/GEECS-BELLA/GEECS-Plugins",
        "commit": "cc35d3e0e7039089a5e449afce1ef50bad6459e6",
        "branch": "main",
        "dirty": false
      },
      "dependencies": {
        "image_analysis": "1.1.0",
        "numpy": "2.0.0",
        "scipy": "1.12.0"
      },
      "config": {
        "scan_analyzer": {
          "class": "Array2DScanAnalyzer",
          "module": "scan_analysis.analyzers.common.array2D_scan_analysis",
          "config": {
            "type": "array2d",
            "device_name": "UC_HiResMagCam",
            "priority": 0,
            "file_tail": ".png"
          }
        },
        "image_analyzer": {
          "class": "BeamAnalyzer",
          "module": "image_analysis.offline_analyzers.beam_analyzer",
          "config": {
            "camera_config_name": "UC_HiResMagCam"
          }
        }
      },
      "notes": "Standard undulator analysis with updated calibration"
    },
    {
      "timestamp": "2026-02-04T15:45:00Z",
      "columns_written": [
        "UC_HiResMagCam peak_energy"
      ],
      "software": {
        "name": "scan_analysis",
        "version": "0.2.0"
      },
      "notes": "Re-ran with corrected energy calibration"
    }
  ]
}
```

### 5.3 YAML Format

The same minimal example in YAML:

```yaml
schema_version: "0.1"
analyses:
  - timestamp: "2026-02-04T20:30:00Z"
    columns_written:
      - centroid_x
      - centroid_y
```

## 6. Versioning

This specification follows semantic versioning for the `schema_version` field:

- **Major version** (1.x → 2.x): Breaking changes that may require migration
- **Minor version** (0.1 → 0.2): Backwards-compatible additions
- **Patch version**: Reserved for clarifications (not reflected in schema_version)

Implementations SHOULD warn (but not fail) when encountering an unknown schema version.

## 7. Implementation Notes

### 7.1 For Python

A reference implementation is provided in the `scan_analysis` package. See `scan_analysis.provenance` module.

### 7.2 For Other Languages

Any language that can read/write JSON can implement this standard. The key requirements are:

1. Read existing provenance file (if present)
2. Append new entry to `analyses` array
3. Write back atomically (using temp file + rename, or file locking)

### 7.3 Minimal LabView Implementation

LabView can implement this by:
1. Building a JSON string manually
2. Appending to the provenance file

```
timestamp = Format Date/Time (ISO 8601)
json = "  {\"timestamp\": \"" + timestamp + "\", \"columns_written\": [\"col1\", \"col2\"]}"
Append to File (provenance_path, json + ",\n")
```

## 8. Future Considerations

The following may be addressed in future versions:

- Provenance chains (analysis A used output from analysis B)
- Data file checksums for integrity verification
- Standard vocabulary for common analysis types
- Machine-readable configuration schemas

---

## Appendix A: JSON Schema

See `schema/v0.1/provenance.schema.json` for a formal JSON Schema definition that can be used for validation.

## Appendix B: Changelog

### Version 0.1 (2026-02-04)
- Initial draft specification
