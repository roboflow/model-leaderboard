#!/bin/bash
set -e

current_path=$(pwd)

folders=(
    "yolov8"
    "yolov9"
    "yolov10"
    "rt-detr"
)

for folder in ${folders[@]}; do
    cd "$current_path/models/object_detection/$folder"

    if [ -f results.json ]; then
        mv results.json results.json.old
    fi

    if [ ! -f results.json ]; then
        # Create virtual environment
        uv venv

        # Define full paths to Python and Pip in venv
        VENV_PY=./.venv/bin/python
        VENV_PIP=./.venv/bin/pip

        # Install dependencies
        $VENV_PIP install -r requirements.txt

        # Force install custom supervision version
        $VENV_PIP install --force-reinstall --no-deps "git+https://github.com/rafaelpadilla/supervision.git@fix/mAP"

        # Run the script inside the venv
        $VENV_PY run.py
    else
        echo "results.json already exists in $folder"
    fi
done
