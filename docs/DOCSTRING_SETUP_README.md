# GEECS Plugin Suite - Docstring Standardization Setup

This document describes the docstring standardization infrastructure that has been set up for the GEECS Plugin Suite.

## What Was Implemented

### 1. Pre-commit Hooks (`.pre-commit-config.yaml`)
- **pydocstyle** - Checks docstring style and completeness using NumPy convention
- **ruff** - Enhanced with docstring rules (D-series) for additional validation
- **Exclusions** - Test files, scripts, and auto-generated code are excluded

### 2. Development Dependencies (`pyproject.toml`)
- **pydocstyle** - Docstring style checker
- **interrogate** - Docstring coverage reporting tool
- **ruff configuration** - Docstring rules enabled with NumPy convention

### 3. Enhanced MkDocs Configuration (`mkdocs.yml`)
- **Better auto-documentation** - Enhanced mkdocstrings settings for improved API docs
- **Signature display** - Shows type annotations and inheritance
- **Cross-referencing** - Better linking between related components

### 4. Documentation Templates (`docs/docstring_templates.md`)
- **NumPy-style templates** - For modules, classes, functions, and properties
- **GEECS-specific examples** - Common patterns for device interfaces, analysis functions
- **Best practices** - Guidelines for writing effective docstrings

### 5. Development Guidelines (`docs/development_guidelines.md`)
- **Comprehensive workflow** - From code writing to documentation generation
- **Quality standards** - What constitutes good documentation
- **Tool usage** - How to use the various quality checking tools
- **Migration strategy** - Phased approach to improving documentation

### 6. Coverage Checking Script (`scripts/check_docstring_coverage.py`)
- **Automated reporting** - Check docstring coverage across all packages
- **Multiple tools** - Runs interrogate, pydocstyle, and ruff checks
- **Package-specific reports** - Individual coverage for each GEECS package

## Getting Started

### 1. Install Dependencies

```bash
# Install the new development dependencies
poetry install

# Install pre-commit hooks
pre-commit install
```

### 2. Check Current Status

```bash
# Run the comprehensive docstring coverage check
python scripts/check_docstring_coverage.py

# Or run individual tools
interrogate .                    # Coverage report
pydocstyle --convention=numpy .  # Style checking
ruff check --select D .         # Ruff docstring rules
```

### 3. Test Pre-commit Hooks

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run only docstring-related hooks
pre-commit run pydocstyle --all-files
```

### 4. Generate Documentation

```bash
# Serve documentation locally to see auto-generated API docs
mkdocs serve

# Build documentation
mkdocs build
```

## Current Configuration

### Docstring Rules
- **Convention**: NumPy style
- **Coverage target**: 80% (configured in interrogate)
- **Exclusions**: Tests, scripts, auto-generated files

### Pre-commit Behavior
- **pydocstyle** runs on all Python files (with exclusions)
- **ruff** includes docstring rules in addition to formatting
- **Automatic fixing** where possible

### Quality Thresholds
- **interrogate**: 80% docstring coverage target
- **pydocstyle**: NumPy convention with gradual adoption (D100, D104 ignored initially)
- **ruff**: Docstring rules enabled but not blocking initially

## Usage Examples

### Writing a New Function

```python
def analyze_beam_profile(image, roi=None):
    """
    Analyze beam profile from camera image.

    Parameters
    ----------
    image : numpy.ndarray
        2D array representing the camera image
    roi : tuple of int, optional
        Region of interest as (top, bottom, left, right)

    Returns
    -------
    dict
        Analysis results containing beam parameters

    Examples
    --------
    >>> image = np.random.rand(100, 100)
    >>> results = analyze_beam_profile(image)
    >>> print(results['centroid_x'])
    """
    # Implementation here
    pass
```

### Adding API Documentation

1. Create a markdown file in the appropriate docs section:

```markdown
# Beam Analysis

::: image_analysis.analyzers.beam_analyzer
    options:
      show_source: false
```

2. Add it to `mkdocs.yml` navigation

### Checking Your Work

```bash
# Before committing
pre-commit run --all-files

# Check specific package
interrogate ImageAnalysis/image_analysis/ -v

# Test documentation build
mkdocs serve
```

## Migration Strategy

### Phase 1: Infrastructure ✅ (Complete)
- Set up all tooling and configuration
- Create templates and guidelines
- Establish quality checking workflow

### Phase 2: Core APIs (Next Steps)
1. **Start with base classes**:
   - `ImageAnalyzer` base class
   - `GEECSScanner` main class
   - Core data structures

2. **Focus on public interfaces**:
   - Main entry points
   - Frequently used functions
   - Configuration classes

3. **Prioritize by usage**:
   - Most commonly used modules first
   - Public APIs before private methods

### Phase 3: Comprehensive Coverage
- Document all public APIs
- Achieve 80%+ coverage across all packages
- Add comprehensive examples

### Phase 4: Enhancement
- Add performance benchmarks
- Create tutorial content
- Enhance cross-referencing

## Troubleshooting

### Common Issues

**Pre-commit hook fails with "command not found"**
```bash
# Make sure dependencies are installed
poetry install
```

**Docstring errors for existing code**
- Start by adding basic docstrings to new code
- Gradually improve existing code
- Use the templates as reference

**Documentation not appearing in mkdocs**
- Check that modules are properly imported
- Verify mkdocs.yml configuration
- Ensure docstrings follow NumPy convention

### Getting Help

1. **Check the templates**: `docs/docstring_templates.md`
2. **Review guidelines**: `docs/development_guidelines.md`
3. **Run coverage check**: `python scripts/check_docstring_coverage.py`
4. **Test locally**: `mkdocs serve`

## Benefits

### Immediate Benefits
- **Automated quality checking** in pre-commit pipeline
- **Consistent documentation** standards across packages
- **Better IDE support** with hover documentation
- **Professional appearance** for the project

### Long-term Benefits
- **Easier onboarding** for new developers
- **Automatic API documentation** generation
- **Reduced maintenance burden** through standardization
- **Better discoverability** of functionality

## Next Steps

1. **Install and test the setup**:
   ```bash
   poetry install
   pre-commit install
   python scripts/check_docstring_coverage.py
   ```

2. **Start documenting new code** using the templates

3. **Gradually improve existing code** following the migration strategy

4. **Monitor progress** using the coverage reporting tools

The infrastructure is now in place to support high-quality, consistent documentation across the entire GEECS Plugin Suite. The tools will help maintain standards automatically while the templates and guidelines provide clear direction for developers.
