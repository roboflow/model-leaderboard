# RTMDet Installation Guide

This guide will help you set up Python 3.9 and install RTMDet along with OpenMIM

## Step 1: Set Up a Virtual Environment

First, ensure you have Python 3.9 installed. You can check this by running:

```sh
python3.9 --version
```

```sh
python3.9 -m venv .venv
source .venv/bin/activate
```

## Step 2: Install requirements

```sh
pip install https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp39-cp39-linux_x86_64.whl
pip install -U openmim
pip install setuptools
mim install "mmengine>=0.6.0"
pip install https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv-2.0.0-cp39-cp39-manylinux1_x86_64.whl
mim install "mmdet>=3.0.0,<4.0.0"
mim install "mmyolo==0.6.0"
```
