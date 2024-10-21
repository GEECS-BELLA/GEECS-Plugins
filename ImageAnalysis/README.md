# ImageAnalysis

Central sub-repository for online and offline analysis of BELLA exeriments' images.

> See the How-To Guide for using ImageAnalysis with GEECS Point Grey Camera Device:  [HTU HTG-200 Python Analysis in GEECS Point Grey Camera Device](https://docs.google.com/document/d/1hNEX6-nev_7Bc3k8dPKlBpR2-aLtuyR2RxZLRwsP-ic/edit?usp=sharing)

> For a tutorial on the coding aspect of implementing ImageAnalysis, see slides:  [GEECS-Python Analysis Tutorial (Coding)](https://docs.google.com/presentation/d/1RU251CXiWsM73NsBJtd_jdQ7tNdZbKSX9x-xBv1vtJw/edit?usp=sharing)

## Install python on device computer

Install python-3.7.9 (32-bit version) on device computer. Select "Add python 3.7.9 to PATH" on first installer window.

To check if python is installed, open Windows Powershell and use command:
`python --version'

Change directory to Z-drive:

`cd Z:`

Change directory to GEECS-Plugins folder:

`cd '.\software\control-all-loasis\HTU\Active Version\GEECS-Plugins\'`

Install Image Analysis package:

`py -3.7-32 -m pip install ./ImageAnalysis`
