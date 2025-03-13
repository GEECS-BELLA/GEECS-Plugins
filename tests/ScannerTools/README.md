# Integration tests for GEECS Scanner tools
Tests for integrating:
* GEECS-Scanner-GUI
* ScanAnalysis

## Environment
1. [Install poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
1. From the directory containing this README, run `poetry install`

## Running tests
Tests can be run in various ways, including:
* *From the command line*. Run `poetry run pytest`
* *From an IDE (e.g. VSCode, PyCharm)*. Make sure the Python interpreter in your 
  IDE is set to the one from the poetry virtual environment (type `poetry env info`
  from the command line to get its path). Then run tests from the test facility 
  of your IDE.
