#!/bin/bash
set -e

# This script is made to be run overnight, producing results.json for every model in the object_detection folder.

current_path=$(pwd)

# Iteration order. Start with quick-running ones to surface errors faster.
folders=(
    "yolov8"
    "yolov9"
    "yolov10"
    "rt-detr"
)

for folder in ${folders[@]}; do
    cd "$current_path/models/object_detection/$folder"
    
    # If results.json exists, move it to results.json.old
    if [ -f results.json ]; then
        mv results.json results.json.old
    fi

    # If results.json doesn't exist, set up and run
    if [ ! -f results.json ]; then
        uv venv
        source .venv/bin/activate
        uv pip install -r requirements.txt
        
        # Force install custom supervision version
        uv pip install --force-reinstall --no-deps "git+https://github.com/rafaelpadilla/supervision.git@fix/mAP"
        
        python run.py
        deactivate
    else
        echo "results.json already exists in $folder"
    fi
done
