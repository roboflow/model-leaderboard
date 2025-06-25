#!/bin/bash
set -e

# This script is made to be run over night, producing results.py for every model in the object_detection folder.
# I hacked it together for initial launch only, but it might be useful as a starting point for later automated solutions.

current_path=$(pwd)

# Iteartion order. Start with quick-running ones to surface errors faster.
folders=(
    "yolov8"
    "yolov9"
    "yolov10"
    "rt-detr"
)


for folder in ${folders[@]}; do
    cd $current_path/models/object_detection/$folder
    # If results.json exists, move it to results.json.old
    if [ -f results.json ]; then
        mv results.json results.json.old
    fi

    # If folder stards with yolov9, use special rules
    if [ ! -f results.json ]; then
        if [[ $folder == yolov9* ]]; then
            bash run_predictions.sh
        fi
        uv venv
        source .venv/bin/activate
        uv pip install -r requirements.txt
        python run.py
        deactivate
    else
        echo "results.json already exists in $folder"
    fi
done
