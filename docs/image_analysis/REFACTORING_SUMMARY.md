# ImageAnalysis Module Refactoring Summary

## Overview
This document summarizes the major refactoring work completed to standardize and simplify the ImageAnalysis module architecture.

**Date**: November 7, 2025
**Scope**: ImageAnalysis module - core types, base classes, and offline analyzers

---

## Key Changes

### 1. New Unified Result Type: `ImageAnalyzerResult`

**Location**: `ImageAnalysis/image_analysis/types.py`

Created a new dataclass `ImageAnalyzerResult` to replace the legacy dictionary-based return format:

```python
@dataclass
class ImageAnalyzerResult:
    """Unified result type for image analysis operations."""
    data_type: str  # "1d" or "2d"
    processed_image: Optional[np.ndarray] = None
    scalars: Dict[str, Union[float, int, str]] = field(default_factory=dict)
    lineouts: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    render_data: Dict[str, Any] = field(default_factory=dict)
```

**Benefits**:
- Type-safe attribute access (e.g., `result.scalars` instead of `result["analyzer_return_dictionary"]`)
- Clear separation of concerns (scalars, lineouts, metadata, render data)
- IDE autocomplete and type checking support
- Simpler to understand for new contributors

---

### 2. Base Class Updates

#### ImageAnalyzer (`ImageAnalysis/image_analysis/base.py`)

**Changes**:
- Updated `analyze_image()` signature to return `ImageAnalyzerResult`
- Maintained backward compatibility with `build_return_dictionary()` for legacy code
- Added compatibility layer in `analyze_image_file()` to handle both old and new return types

**Migration Path**:
```python
# Old way (still supported via build_return_dictionary)
def analyze_image(self, image, auxiliary_data=None) -> dict:
    return self.build_return_dictionary(
        return_image=processed,
        return_scalars={"mean": 42.0}
    )

# New way (recommended)
def analyze_image(self, image, auxiliary_data=None) -> ImageAnalyzerResult:
    return ImageAnalyzerResult(
        data_type="2d",
        processed_image=processed,
        scalars={"mean": 42.0}
    )
```

#### StandardAnalyzer (`ImageAnalysis/image_analysis/offline_analyzers/standard_analyzer.py`)

- Updated to return `ImageAnalyzerResult`
- Refactored `analyze_image()` to use new result type
- Simplified scalar computation logic

#### BeamAnalyzer (`ImageAnalysis/image_analysis/offline_analyzers/beam_analyzer.py`)

- Updated to return `ImageAnalyzerResult`
- Added `visualize()` method for simplified rendering
- Created static `render_image()` method for customizable visualization

#### Standard1DAnalyzer (`ImageAnalysis/image_analysis/offline_analyzers/standard_1d_analyzer.py`)

- Updated to return `ImageAnalyzerResult`
- Simplified lineout storage using `result.lineouts` dict
- Added `render_1d()` static method for plotting

---

### 3. Offline Analyzers Updated

All offline analyzers were migrated to use `ImageAnalyzerResult`:

| Analyzer | File | Notes |
|----------|------|-------|
| **HiResMagCamAnalyzer** | `Undulator/hi_res_mag_cam_analyzer.py` | Added bowtie fit overlay rendering |
| **ACaveMagCam3ImageAnalyzer** | `Undulator/ACaveMagCam3.py` | Simplified ROI statistics |
| **PhaseDownrampProcessor** | `density_from_phase_analysis.py` | Maintained shock analysis workflow |
| **HASOHimgHasProcessor** | `HASO_himg_has_processor.py` | WaveKit phase computation |

---

### 4. New Rendering Utilities

**Location**: `ImageAnalysis/image_analysis/tools/rendering.py`

Created a composable rendering system with a generic line overlay function:

**Core Functions**:

1. **`base_render_image()`** - Base 2D image rendering with consistent defaults
2. **`add_line_overlay()`** - Generic function for horizontal/vertical line overlays
3. **`add_xy_projections()`** - Convenience wrapper for projection overlays (uses `add_line_overlay()`)
4. **`add_marker()`** - Add markers/dots at specific positions

**Key Innovation: `add_line_overlay()`**

```python
def add_line_overlay(
    ax: Axes,
    lineout: np.ndarray,
    direction: Literal["horizontal", "vertical", "h", "v"],
    scale: float = 0.3,
    offset: float = 0.0,
    color: str = "cyan",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    normalize: bool = True,
    clip_positive: bool = False,
    label: Optional[str] = None,
) -> None:
    """Generic line overlay in any direction."""
```

**Benefits**:
- **Single implementation** for both horizontal and vertical line overlays
- **Eliminates code duplication** (~60 lines of redundant code removed)
- **Easy to add multiple lines** at different positions
- **Flexible placement** via `offset` parameter
- **Consistent behavior** for all line types (projections, fits, derived quantities)
- Support for both standalone figures and subplot integration

**Example Usage**:
```python
# Add horizontal lineout (e.g., bowtie fit weights)
add_line_overlay(ax, lineout, direction='horizontal', offset=0)

# Add vertical lineout on right side
add_line_overlay(ax, lineout, direction='vertical', offset=img_width)

# Add multiple lines at different positions
for i, line in enumerate(lineouts):
    add_line_overlay(ax, line, 'h', offset=i*50, color=colors[i])
```

---

## Migration Guide for Developers

### For New Analyzers

Use `ImageAnalyzerResult` from the start:

```python
from image_analysis.base import ImageAnalyzer
from image_analysis.types import ImageAnalyzerResult

class MyAnalyzer(ImageAnalyzer):
    def analyze_image(self, image, auxiliary_data=None) -> ImageAnalyzerResult:
        # Your processing here
        processed = self.process(image)

        return ImageAnalyzerResult(
            data_type="2d",
            processed_image=processed,
            scalars={"metric": 123.45},
            metadata=auxiliary_data or {}
        )
```

### For Existing Analyzers (Gradual Migration)

1. Keep using `build_return_dictionary()` initially
2. Update return type hint to `ImageAnalyzerResult`
3. Replace `build_return_dictionary()` with direct `ImageAnalyzerResult` construction
4. Test thoroughly with existing workflows

### Accessing Result Data

```python
# Old dictionary access (avoid in new code)
mean_value = result["analyzer_return_dictionary"]["mean"]

# New attribute access (recommended)
mean_value = result.scalars["mean"]
processed_img = result.processed_image
metadata = result.metadata
```

---

## Backward Compatibility

The refactoring maintains backward compatibility:

1. **Legacy dictionary returns**: Still work via `build_return_dictionary()`
2. **File I/O**: `analyze_image_file()` handles both old and new formats
3. **External integrations**: ScanAnalysis and other modules not yet updated will continue working

---

## Benefits Summary

1. **Type Safety**: IDE support, fewer runtime errors
2. **Clarity**: Clear separation of scalars, lineouts, metadata
3. **Consistency**: Unified interface across all analyzers
4. **Simplicity**: Easier for novice contributors to understand
5. **Maintainability**: Centralized rendering and utilities
6. **Extensibility**: Easy to add new result fields (e.g., `render_data`)

---

## Next Steps

1. ✅ Update offline analyzers to `ImageAnalyzerResult`
2. ⏳ Update ScanAnalysis to use new result type
3. ⏳ Add comprehensive examples in documentation
4. ⏳ Consider deprecation warnings for old dictionary methods
5. ⏳ Full migration of all remaining analyzers

---

## Files Modified

### Core Types
- `ImageAnalysis/image_analysis/types.py` - Added `ImageAnalyzerResult`

### Base Classes
- `ImageAnalysis/image_analysis/base.py` - Updated `ImageAnalyzer`
- `ImageAnalysis/image_analysis/offline_analyzers/standard_analyzer.py`
- `ImageAnalysis/image_analysis/offline_analyzers/beam_analyzer.py`
- `ImageAnalysis/image_analysis/offline_analyzers/standard_1d_analyzer.py`

### Offline Analyzers
- `ImageAnalysis/image_analysis/offline_analyzers/Undulator/hi_res_mag_cam_analyzer.py`
- `ImageAnalysis/image_analysis/offline_analyzers/Undulator/ACaveMagCam3.py`
- `ImageAnalysis/image_analysis/offline_analyzers/density_from_phase_analysis.py`
- `ImageAnalysis/image_analysis/offline_analyzers/HASO_himg_has_processor.py`

### New Utilities
- `ImageAnalysis/image_analysis/tools/rendering.py` - Centralized rendering functions

### Documentation
- `docs/image_analysis/REFACTORING_SUMMARY.md` - This document
