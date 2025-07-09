# LW-DETR Installation Guide

This guide will help you set up Python environment and install LW-DETR.

## Step 1: Set Up a Virtual Environment

First, ensure you have Python 3 installed. You can check this by running and following comamnd and create a virtual environment:

```sh
python3 venv .venv
source .venv/bin/activate
```

## Step 2: Clone LW-DETR repository

In order to use the models, we need to clone the original LW-DETR repository in the path: 'models/object_detection/lw-detr'. After you are in the path, run in your shell:

```sh
git clone https://github.com/Atten4Vis/LW-DETR.git
```

## Step 3: Install requirements

LW-DETR provides some installs that should be adapted to your hardware. You will need to specify the CUDA version of the torch install and find a version compatible with LW-DETR.
In Google Colab to the date 8 of July of 2025 the installs are:
```sh

pip install -r requirements.txt

cd LW-DETR/models/ops

pip install torch==2.5.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -U git+https://github.com/qubvel/transformers@fix-custom-kernels

python3 setup.py build install
```

After installing everything return to the path 'models/object_detection/lw-detr' and you are ready to run run.py
