#!/usr/bin/env python3
"""
Script to check docstring coverage across GEECS plugin packages.

This script provides a convenient way to check docstring coverage
for all GEECS packages and generate reports.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """
    Run a command and return the result.

    Parameters
    ----------
    cmd : list
        Command to run as a list of strings
    description : str
        Description of what the command does

    Returns
    -------
    bool
        True if command succeeded, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("ERROR: Command not found. Make sure the tool is installed.")
        return False


def check_package_coverage(package_path):
    """
    Check docstring coverage for a specific package.

    Parameters
    ----------
    package_path : Path
        Path to the package directory

    Returns
    -------
    bool
        True if coverage check succeeded, False otherwise
    """
    if not package_path.exists():
        print(f"Package path {package_path} does not exist, skipping...")
        return True

    return run_command(
        ["interrogate", str(package_path), "-v"],
        f"Checking docstring coverage for {package_path.name}",
    )


def main():
    """Run all docstring coverage checks."""
    print("GEECS Plugin Suite - Docstring Coverage Report")
    print("=" * 60)

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # List of packages to check (excluding third-party and non-core packages)
    packages = [
        "GEECS-Scanner-GUI/geecs_scanner",
        "ImageAnalysis/image_analysis",
        "ScanAnalysis/scan_analysis",
        "GEECS-Data-Utils/geecs_data_utils",
        "GEECS-PythonAPI/geecs_python_api",
    ]

    success_count = 0
    total_count = 0

    # Check overall project coverage first
    print("\n" + "=" * 60)
    print("OVERALL PROJECT COVERAGE")
    print("=" * 60)

    if run_command(
        ["interrogate", str(project_root), "-v", "--ignore-module"],
        "Overall project docstring coverage",
    ):
        success_count += 1
    total_count += 1

    # Check each package individually
    for package in packages:
        package_path = project_root / package
        if check_package_coverage(package_path):
            success_count += 1
        total_count += 1

    # Run pydocstyle checks
    print("\n" + "=" * 60)
    print("PYDOCSTYLE CHECKS")
    print("=" * 60)

    if run_command(
        ["pydocstyle", "--convention=numpy", str(project_root)],
        "Checking docstring style compliance",
    ):
        success_count += 1
    total_count += 1

    # Run ruff docstring checks
    print("\n" + "=" * 60)
    print("RUFF DOCSTRING CHECKS")
    print("=" * 60)

    if run_command(
        ["ruff", "check", "--select", "D", str(project_root)],
        "Checking docstring rules with Ruff",
    ):
        success_count += 1
    total_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful checks: {success_count}/{total_count}")

    if success_count == total_count:
        print("✅ All docstring checks passed!")
        return 0
    else:
        print("❌ Some docstring checks failed. See output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
