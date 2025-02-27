#!/bin/bash

###############################################################################
# FUNCTIONS
###############################################################################

check_poetry_env() {
    local lock_file="poetry.lock"
    local hash_file=".poetry-lock-hash"

    # check if lock file exists
    if [ ! -f "$lock_file" ]; then
        echo "File 'poetry.lock' not found."
        return 1
    fi

    # get current lock file hash
    local current_hash=$(md5sum "$lock_file" | awk '{ print $1 }')

    # if hash file does not exist or current hash does not equal hash file
    if [ ! -f "$hash_file" ] || [ "$current_hash" != "$(cat "$hash_file")" ]; then
        echo "Poetry environment not up to date. Running poetry install..."
        poetry install
        echo "$current_hash" > "$hash_file"
        return 0
    else
        echo "Poetry environment is up to date"
        return 0
    fi
}

###############################################################################
# GLOBAL VARIABLES
###############################################################################

#BASE_PATH="C:/GEECS/Developers Version/source/GEECS-Plugins/ScanAnalysis"
BASE_PATH='Z:/software/control-all-loasis/HTU/Active Version/GEECS-Plugins/ScanAnalysis'

GUI_PATH=$BASE_PATH"/live_watch/scan_analysis_gui"

###############################################################################
# MAIN FUNCTION
###############################################################################

main() {
    # go to expected poetry location
    cd "$BASE_PATH"

    # run check_poetry_env, exit script if no poetry.lock found
    if ! check_poetry_env; then
        echo "Failed to check/update Poetry environment."
        exit 1
    fi

    # go to gui location, run gui script
    cd "$GUI_PATH"
    poetry run python main.py
}

###############################################################################
# EXECUTE MAIN
###############################################################################

main "$@"
