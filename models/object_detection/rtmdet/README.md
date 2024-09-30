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

```sh
pip install openmim
mim install -r requirements.txt
```
