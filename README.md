# GEECS-Plugins
Add-ons for the Generalized Equipment and Experiment Control system-

## Using Pre-Commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to automatically run formatting, linting, and other checks before you make a commit.

### 1. Install `pre-commit`
```bash
pip install pre-commit
```

### 2. Install the hooks

Run this once in the repo root to set up the Git hook scripts:
```bash
pre-commit install
```

Now, every time you run git commit, pre-commit will:

- Run the configured checks on staged files.
- Stop the commit if any check fails (you’ll need to fix the issues and re-stage).
- The configuration is in .pre-commit-config.yaml in the repo root.

## Documentation 
https://sites.google.com/a/lbl.gov/geecs/plugins?authuser=0 (original)

read the docs: https://geecs-plugins.readthedocs.io/en/latest/

## License

*** Copyright Notice ***

“GEECS (Generalized Equipment and Experiment Control System)”, Copyright (c) 2016, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

 

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation & Partnerships Office at  IPO@lbl.gov.

 

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.

 
