#!/bin/bash

BASE_PATH='C:/GEECS/Developers Version/source/GEECS-Plugins/ScanAnalysis'
#BASE_PATH='Z:/software/control-all-loasis/HTU/Active Version/GEECS-Plugins/ScanAnalysis'

cd "$BASE_PATH"
poetry install
cd "live_watch/scan_analysis_gui/"
poetry run python main.py
