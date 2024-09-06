#!/bin/bash

wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt -O yolov9m-conv.pt

git clone git@github.com:WongKinYiu/yolov9.git
cd yolov9
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python detect.py --source "../../../../data/coco-dataset/test/images/" --img 640 --device cpu --weights '../yolov9m-conv.pt' --name detection_out --save-txt --save-conf
deactivate
cd ..
