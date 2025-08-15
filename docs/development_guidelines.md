# GEECS Plugin Suite - Development Guidelines

This document outlines the development standards and best practices for the GEECS Plugin Suite, with a focus on documentation and code quality.

## Documentation Standards

### Docstring Requirements

All public modules, classes, functions, and methods must have docstrings following the NumPy convention. This ensures:

- Consistent documentation across all GEECS plugins
- Automatic API documentation generation via mkdocstrings
- Better IDE support and developer experience
- Professional documentation that matches scientific software standards

### What Needs Documentation

#### Required Docstrings
- **All public modules** - Top-level description of module purpose
- **All public classes** - Class purpose, parameters, attributes, methods overview
- **All public functions/methods** - Purpose, parameters, returns, exceptions
- **All public properties** - What the property represents and returns

#### Optional but Recommended
- **Private methods** - If they contain complex logic
- **Configuration classes** - Even if simple, for clarity
- **Exception classes** - Custom exceptions should explain when they're raised

### Docstring Quality Standards

#### Minimum Requirements
- One-line summary (clear and descriptive, under 79 characters)
- Parameters section with types and descriptions
- Returns section with type and description
- Examples section for non-trivial functions

#### Relaxed Style Rules
To balance code quality with developer productivity, we've disabled some overly restrictive pydocstyle rules:
- **D401** (Imperative mood) - Both "Returns the result" and "Return the result" are acceptable
- **D200** (One-line format) - Multi-line docstrings can be more flexible in formatting
- **D400** (Period requirement) - Periods at the end of summaries are recommended but not required

#### Best Practices
- Include realistic, runnable examples
- Document side effects and performance considerations
- Reference related functions/classes in "See Also"
- Note GEECS-specific integration details
- Include thread safety information when relevant

## Code Quality Tools

### Pre-commit Hooks

Our pre-commit configuration automatically checks:

- **pydocstyle** - Docstring style and completeness (NumPy convention)
- **ruff** - Code formatting and docstring rules
- **Standard hooks** - Trailing whitespace, file endings, syntax validation

### Running Quality Checks

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks on all files
pre-commit run --all-files

# Check docstring coverage
interrogate .

# Run specific docstring checks
pydocstyle geecs_scanner/

# Run ruff with docstring rules
ruff check --select D .
```

### Configuration

All tools are configured in `pyproject.toml`:

- **Ruff** - Docstring rules enabled with NumPy convention
- **pydocstyle** - NumPy convention with gradual adoption settings
- **interrogate** - Docstring coverage reporting with 80% target

## Development Workflow

### 1. Before Writing Code

- Review existing code patterns in the relevant package
- Check the docstring templates in `docs/docstring_templates.md`
- Understand the module's role in the GEECS ecosystem

### 2. Writing Code

- Write the function/class signature first
- Add a basic docstring immediately
- Implement the functionality
- Enhance the docstring with examples and details

### 3. Before Committing

- Run `pre-commit run --all-files` to check quality
- Ensure all new public APIs have complete docstrings
- Test that examples in docstrings actually work
- Verify documentation builds correctly with `mkdocs serve`

### 4. Code Review Checklist

- [ ] All public APIs have NumPy-style docstrings
- [ ] Examples are realistic and runnable
- [ ] Parameter types and descriptions are accurate
- [ ] Return values are documented
- [ ] Exceptions are documented where appropriate
- [ ] GEECS-specific integration notes are included

## Package-Specific Guidelines

### GEECS Scanner GUI
- Document GUI components and their interactions
- Include examples of programmatic usage
- Note threading considerations for GUI operations
- Document configuration file formats

### Image Analysis
- Document image processing algorithms clearly
- Include mathematical descriptions where appropriate
- Provide examples with sample data
- Document performance characteristics

### Scan Analysis
- Document analysis algorithms and their assumptions
- Include references to scientific papers where applicable
- Provide examples with realistic scan data
- Document data format requirements

### GEECS Python API
- Document device interfaces thoroughly
- Include connection and error handling examples
- Document timing and synchronization requirements
- Provide troubleshooting guidance

## Documentation Generation

### Local Development

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build

# Check for broken links
mkdocs build --strict
```

### Auto-generated API Documentation

Our mkdocstrings configuration automatically generates API documentation from docstrings. To include a new module:

1. Add the module to the appropriate section in `mkdocs.yml`
2. Create a markdown file that imports the module:

```markdown
# Module Name

::: package.module
    options:
      show_source: false
```

### Documentation Structure

```
docs/
├── index.md                    # Main landing page
├── docstring_templates.md      # Templates and examples
├── development_guidelines.md   # This file
├── installation.md            # General installation
├── geecs_scanner/             # Scanner GUI docs
├── image_analysis/            # Image analysis docs
├── scan_analysis/             # Scan analysis docs
└── geecs_python_api/          # Python API docs
```

## Common Pitfalls and Solutions

### Docstring Issues

**Problem**: "Missing docstring in public module"
**Solution**: Add module-level docstring at the top of the file

**Problem**: "One-line docstring should fit on one line"
**Solution**: Keep summary under 79 characters, use imperative mood

**Problem**: "No blank line after section header"
**Solution**: Always add blank line after section headers like "Parameters"

### Integration Issues

**Problem**: Documentation not appearing in mkdocs
**Solution**: Check that the module is properly imported and referenced

**Problem**: Examples not rendering correctly
**Solution**: Use proper doctest format with `>>>` prompts

**Problem**: Type annotations not showing
**Solution**: Ensure `show_signature_annotations: true` in mkdocs.yml

## Migration Strategy

### Phase 1: Infrastructure (Complete)
- ✅ Set up pre-commit hooks
- ✅ Configure quality tools
- ✅ Create templates and guidelines
- ✅ Enhance mkdocs configuration

### Phase 2: Core APIs (Next)
- Document main entry points and base classes
- Focus on public interfaces
- Prioritize frequently used functions

### Phase 3: Comprehensive Coverage
- Document all public APIs
- Add examples to all functions
- Achieve 80%+ docstring coverage

### Phase 4: Enhancement
- Add more detailed examples
- Include performance benchmarks
- Create tutorial content

## Getting Help

### Resources
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [mkdocstrings Documentation](https://mkdocstrings.github.io/)
- [pydocstyle Error Codes](http://www.pydocstyle.org/en/stable/error_codes.html)

### Team Support
- Check existing code for patterns
- Review `docs/docstring_templates.md` for examples
- Ask for code review feedback on documentation
- Use the pre-commit hooks to catch issues early

## Continuous Improvement

This documentation standard will evolve with the project. Suggestions for improvements should be:

1. Discussed with the team
2. Tested on a small scale first
3. Updated in this guidelines document
4. Communicated to all developers

Remember: Good documentation is an investment in the future maintainability and usability of the GEECS Plugin Suite.
