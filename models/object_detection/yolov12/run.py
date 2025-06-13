import argparse
import sys
from pathlib import Path
from typing import List, Optional

import supervision as sv
from supervision.metrics import F1Score, MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import DATASET_DIR
from utils import (
    download_file,
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
)

MODEL_URLS: dict[str, str] = {
    "yolov12n.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12n.pt",
    "yolov12s.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12s.pt",
    "yolov12m.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12m.pt",
    "yolov12l.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12l.pt",
    "yolov12x.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt",
}


LICENSE = "APGL-3.0"
RUN_PARAMETERS = dict(
    imgsz=640,
    iou=0.6,
    max_det=300,
    conf=0.001,
    verbose=False,
)
import os
os.environ["FLASH_ATTN_DISABLED"] = "1"

GIT_REPO_URL = "https://github.com/sunsmarterjie/yolov12"
PAPER_URL = "https://arxiv.org/abs/2502.12524"


def run_on_image(model, image) -> sv.Detections:
    result = model(image, **RUN_PARAMETERS)[0]
    detections = sv.Detections.from_ultralytics(result)
    return detections


def run(
    model_ids: List[str],
    skip_if_result_exists=False,
    dataset: Optional[sv.DetectionDataset] = None,
) -> None:
    """
    Run the evaluation for the given models and dataset.

    Arguments:
        model_ids: List of model ids to evaluate. Evaluate all models if None.
        skip_if_result_exists: If True, skip the evaluation if the result json already exists.
        dataset: If provided, use this dataset for evaluation. Otherwise, load the dataset from the default directory.
    """  # noqa: E501 // docs
    if not model_ids:
        model_ids = MODEL_URLS.keys()

    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")

        print("Downloading model...")
        if not Path(model_id).exists():
            download_file(MODEL_URLS[model_id], model_id)
            print(f"Model {model_id} downloaded!")
        else:
            print(f"Model {model_id} already exists!")

        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")
            continue

        if dataset is None:
            dataset = load_detections_dataset(DATASET_DIR)

        model = YOLO(model_id)

        predictions = []
        targets = []
        print("Evaluating...")
        for _, image, target_detections in tqdm(dataset, total=len(dataset)):
            # Run model
            detections = run_on_image(model, image)
            predictions.append(detections)
            targets.append(target_detections)

        mAP_metric = MeanAveragePrecision()
        f1_score = F1Score()

        f1_score_result = f1_score.update(predictions, targets).compute()
        mAP_result = mAP_metric.update(predictions, targets).compute()

        write_result_json(
            model_id=model_id,
            model_name=model_id,
            model=model,
            mAP_result=mAP_result,
            f1_score_result=f1_score_result,
            license_name=LICENSE,
            run_parameters=RUN_PARAMETERS,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_ids",
        nargs="*",
        help="Model ids to evaluate. If not provided, evaluate all models.",
    )
    parser.add_argument(
        "--skip_if_result_exists",
        action="store_true",
        help="If specified, skip the evaluation if the result json already exists.",
    )
    args = parser.parse_args()

    run(args.model_ids, args.skip_if_result_exists)
