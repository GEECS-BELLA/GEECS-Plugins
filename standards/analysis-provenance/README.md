# Analysis Provenance Standard.

A lightweight standard for tracking the provenance of derived data columns in analysis files.

## Quick Start

When your analysis tool writes new columns to a data file, create/update a provenance file alongside it:

**Data file:** `s123.txt`
**Provenance file:** `s123.provenance.json`

### Minimal Example

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

### Full Example (with reproducibility info)

```json
{
  "schema_version": "0.1",
  "analyses": [
    {
      "timestamp": "2026-02-04T14:30:00Z",
      "columns_written": ["peak_energy", "charge"],
      "software": {
        "name": "scan_analysis",
        "version": "0.2.0"
      },
      "code_version": {
        "repository": "https://github.com/GEECS-BELLA/GEECS-Plugins",
        "commit": "cc35d3e0e7039089a5e449afce1ef50bad6459e6",
        "dirty": false
      },
      "config": {
        "analyzer_class": "Array2DScanAnalyzer",
        "device_name": "UC_HiResMagCam"
      }
    }
  ]
}
```

## Why?

- **Reproducibility**: Know exactly what code produced each derived value
- **Traceability**: Track which tools modified which columns over time
- **Debugging**: Understand why results changed ("oh, the calibration was updated")

## Documentation

- [Full Specification](SPECIFICATION.md) - Complete standard definition
- [JSON Schema](schema/v0.1/provenance.schema.json) - For validation
- [Examples](schema/v0.1/examples/) - Minimal and full examples

## Implementations

### Python (Reference Implementation)

```python
from scan_analysis.provenance import log_provenance

# After writing columns to your data file:
log_provenance(
    data_file="path/to/s123.txt",
    columns_written=["centroid_x", "centroid_y"],
    software_name="my_analysis",
    software_version="1.0.0"
)
```

### Other Languages

Any language that can write JSON can implement this standard. See [SPECIFICATION.md](SPECIFICATION.md) Section 7 for implementation notes.

## Version

Current: **0.1** (Draft/Experimental)

This is an early version designed for initial adoption and feedback. Breaking changes may occur before 1.0.
