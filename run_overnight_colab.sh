#!/bin/bash
set -e

current_path=$(pwd)

folders=(
    'lw-detr'
    "d-fine"
    "yolov12"
    'rf-detr'
    "yolov9"
    "rt-detr"
    "rtmdet"
    "deim"
    "yolov8"
    "yolov10"
    "yolov11"

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
        VENV_MIM="$venv_dir/bin/mim"

        # Upgrade pip
        $VENV_PY -m pip install --upgrade pip

        # Install dependencies
        $VENV_PIP install -r requirements.txt

        # Override supervision version
        $VENV_PIP install --force-reinstall --no-deps "git+https://github.com/rafaelpadilla/supervision.git@fix/mAP"
        if [[ $folder == yolov12* ]]; then
            $VENV_PIP install --force-reinstall  ultralytics==8.3.151
        fi
        export MPLBACKEND=Agg
        if [[ $folder == rtmdet* ]]; then
            $VENV_PIP install -U openmim

            $VENV_PY -m mim install mmcv==2.0.0
        fi
        if [[ $folder == rf-detr* ]]; then
            $VENV_PIP uninstall -y onnxruntime onnxruntime-gpu
            $VENV_PIP install onnxruntime-gpu

        fi
        if [[ $folder == lw-detr* ]]; then
            if [ ! -d "LW-DETR" ] ; then
                git clone https://github.com/Atten4Vis/LW-DETR.git
            fi
            cd LW-DETR/models/ops

            $VENV_PIP install torch==2.5.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
            $VENV_PIP install -U git+https://github.com/qubvel/transformers@fix-custom-kernels

            $VENV_PY setup.py build install
            cd ../../..
        fi

        # Run script
        $VENV_PY run.py
        # Copy only .json files to Drive
        find "$current_path/models/object_detection/$folder" -name "*.json" -exec cp --parents {} /content/drive/MyDrive/ \;
    else
        echo "results.json already exists in $folder"
    fi

done
