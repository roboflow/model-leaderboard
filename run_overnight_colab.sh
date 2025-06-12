#!/bin/bash
set -e

current_path=$(pwd)

folders=(
    "rtmdet"    
    "deim"
    "yolov8" 
    "yolov10" 
    "yolov11" 
    "yolov12"   
    "yolov9" # ran
    "d-fine"  # ran
    "rt-detr" # ran

)

for folder in ${folders[@]}; do
    cd "$current_path/models/object_detection/$folder"

    if [ -f results.json ]; then
        mv results.json results.json.old
    fi

    if [ ! -f results.json ]; then
        # Create virtual environment using virtualenv
        virtualenv .venv
        project_dir="$current_path/models/object_detection/$folder"

        # Define full paths
        venv_dir="$project_dir/.venv"

        VENV_PY="$venv_dir/bin/python"
        VENV_PIP="$venv_dir/bin/pip"
        VENV_MIM="$venv_dir/bin/mim"  # << fix

        # Upgrade pip
        $VENV_PY -m pip install --upgrade pip

        # Install dependencies
        $VENV_PIP install -r requirements.txt

        # Override supervision version
        $VENV_PIP install --force-reinstall --no-deps "git+https://github.com/rafaelpadilla/supervision.git@fix/mAP"
        export MPLBACKEND=Agg
        $VENV_PIP install -U openmim

        $VENV_MIM install mmcv==2.0.0
        #git clone https://github.com/open-mmlab/mmcv.git
        #cd mmcv
        #MMCV_WITH_OPS=1 $VENV_PIP install --editable . --use-pep517
        #cd ..
        # Run script
        $VENV_PY run.py
        # Copy only .json files to Drive
        find "$current_path/models/object_detection/$folder" -name "*.json" -exec cp --parents {} /content/drive/MyDrive/ \;
    else
        echo "results.json already exists in $folder"
    fi

done
