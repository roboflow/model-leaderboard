# RTMDet Installation Guide

This guide will help you set up Python 3.9 and install RTMDet along with OpenMIM

## Step 1: Set Up a Virtual Environment

First, ensure you have Python 3.9 installed. You can check this by running and following comamnd and create virtualenvironments:

```sh
python3.9 --version
python3.9 -m venv .venv
source .venv/bin/activate
```

## Step 2: Install requirements

We will install the required libraries from requirements.txt, which includes openmim, that gives you the mim command.
Beware that this might be a problematic install, so be sure that the installed torch and openmim versions match your hardware.
```sh
pip install -r requirements.txt
mim install mmcv==2.0.0
```
