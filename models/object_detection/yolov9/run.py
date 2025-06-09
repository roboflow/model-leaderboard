import argparse
import sys
from pathlib import Path
from typing import List, Optional

import supervision as sv
from supervision.metrics import F1Score, MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
)

MODEL_IDS = ["yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e"]
DATASET_DIR = "../../../data/coco-val-2017"
LICENSE = "GPL-3.0"
GIT_REPO_URL = "https://github.com/WongKinYiu/yolov9"
PAPER_URL = "https://arxiv.org/abs/2402.13616"
RUN_PARAMETERS = dict(imgsz=640, conf=0.001, verbose=False)

def run_on_image(model, image) -> sv.Detections:
    result = model.predict(image, **RUN_PARAMETERS)[0]
    detections = sv.Detections.from_ultralytics(result)
    return detections

def run(
    model_ids: List[str],
    skip_if_result_exists=False,
    dataset: Optional[sv.DetectionDataset] = None,
) -> None:
    if not model_ids:
        model_ids = MODEL_IDS

    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")
        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")
            continue

        if dataset is None:
            dataset = load_detections_dataset(DATASET_DIR)

        print(f"Loading model {model_id}...")
        model = YOLO(f"{model_id}.pt")  # descarga automática si está disponible :contentReference[oaicite:1]{index=1}

        predictions, targets = [], []
        print("Running inference on dataset...")
        for _, image, target in tqdm(dataset, total=len(dataset)):
            detections = run_on_image(model, image)
            predictions.append(detections)
            targets.append(target)

        mAP_metric = MeanAveragePrecision()
        f1_score = F1Score()
        f1 = f1_score.update(predictions, targets).compute()
        mAP = mAP_metric.update(predictions, targets).compute()

        write_result_json(
            model_id=model_id,
            model_name=model_id,
            model_git_url=GIT_REPO_URL,
            paper_url=PAPER_URL,
            model=model,
            mAP_result=mAP,
            f1_score_result=f1,
            license_name=LICENSE,
            run_parameters=RUN_PARAMETERS,
        )

