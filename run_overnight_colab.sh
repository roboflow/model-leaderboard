#!/bin/bash
set -e

current_path=$(pwd)

folders=(
    "rt-detr"
    "rtmdet"    
    "deim"
    "yolov8" 
    "yolov10" 
    "yolov11" 
    "yolov12"   
    "yolov9" # ran
    "d-fine"  # ran
)

for folder in ${folders[@]}; do
    cd "$current_path/models/object_detection/$folder"

    if [ -f results.json ]; then
        mv results.json results.json.old
    fi

    if [ ! -f results.json ]; then
        # Create virtual environment using virtualenv
        virtualenv .venv

        # Define full paths
        VENV_PY=./.venv/bin/python
        VENV_PIP=./.venv/bin/pip

        # Upgrade pip
        $VENV_PY -m pip install --upgrade pip

        # Install dependencies
        $VENV_PIP install -r requirements.txt

        # Override supervision version
        $VENV_PIP install --force-reinstall --no-deps "git+https://github.com/rafaelpadilla/supervision.git@fix/mAP"
        export MPLBACKEND=Agg

        # Run script
        $VENV_PY run.py
        cp -r "$current_path/models/object_detection/$folder" /content/driveMyDrive/
    
    else
        echo "results.json already exists in $folder"
    fi

done
