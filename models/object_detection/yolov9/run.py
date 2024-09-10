import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import CLASS_NAMES
from utils import (
    download_file,
    load_detections_dataset,
    result_json_already_exists,
    run_shell_command,
    write_result_json,
)

MODEL_DICT = {
    "yolov9t": {
        "model_url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt",
        "model_filename": "yolov9t-converted.pt",
        "model_run_dir": "yolov9t-out",
    },
    "yolov9s": {
        "model_url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt",
        "model_filename": "yolov9s-converted.pt",
        "model_run_dir": "yolov9s-out",
    },
    "yolov9m": {
        "model_url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt",
        "model_filename": "yolov9m-converted.pt",
        "model_run_dir": "yolov9m-out",
    },
    "yolov9c": {
        "model_url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt",
        "model_filename": "yolov9c-converted.pt",
        "model_run_dir": "yolov9c-out",
    },
    "yolov9e": {
        "model_url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt",
        "model_filename": "yolov9e-converted.pt",
        "model_run_dir": "yolov9e-out",
    },
}  # noqa: E501 // docs
DATASET_DIR = "../../../data/coco-val-2017"
CONFIDENCE_THRESHOLD = 0.001
REPO_URL = "git@github.com:WongKinYiu/yolov9.git"
DEVICE = "0" if torch.cuda.is_available() else "cpu"


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
        model_ids = list(MODEL_DICT.keys())

    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")
        model_values = MODEL_DICT[model_id]

        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")
            continue

        if not Path("yolov9-repo").is_dir():
            run_shell_command(["git", "clone", REPO_URL, "yolov9-repo"])
        download_file(model_values["model_url"], model_values["model_filename"])

        # Make predictions
        shutil.rmtree(
            f"yolov9-repo/runs/detect/{model_values['model_run_dir']}",
            ignore_errors=True,
        )
        run_shell_command(
            [
                "python",
                "detect.py",
                "--source",
                "../../../../data/coco-val-2017/images/val2017",
                "--img",
                "640",
                "--device",
                DEVICE,
                "--weights",
                f"../{model_values['model_filename']}",
                "--name",
                model_values["model_run_dir"],
                "--save-txt",
                "--save-conf",
            ],
            working_directory="yolov9-repo",
        )
        predictions_dict = load_predictions_dict(
            Path(f"yolov9-repo/runs/detect/{model_values['model_run_dir']}")
        )

        if dataset is None:
            dataset = load_detections_dataset(DATASET_DIR)

        predictions = []
        targets = []
        for image_path, _, target_detections in tqdm(dataset, total=len(dataset)):
            # Load predictions
            detections = predictions_dict[Path(image_path).name]
            detections.class_id = np.array(
                [CLASS_NAMES[class_id] for class_id in detections.class_id]
            )
            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]

            predictions.append(detections)
            targets.append(target_detections)

        mAP_metric = sv.metrics.MeanAveragePrecision()
        mAP_result = mAP_metric.update(predictions, targets).compute()
        model = YOLO(model_id)

        write_result_json(
            model_id=model_id, model_name=model_id, model=model, mAP_result=mAP_result
        )


def load_predictions_dict(run_dir: Path) -> Dict[str, sv.Detections]:
    print(f"Loading predictions dataset from {run_dir}...")
    image_dir = run_dir
    labels_dir = run_dir / "labels"
    dataset = {}
    for image_path in image_dir.glob("*.jpg"):
        label_path = labels_dir / (image_path.stem + ".txt")
        detections = labels_to_detections(label_path)
        dataset[image_path.name] = detections
    return dataset


def labels_to_detections(label_path: Path) -> sv.Detections:
    if not label_path.exists():
        print(f"Label file {label_path} not found.")
        return sv.Detections.empty()

    with open(label_path, "r") as f:
        lines = f.readlines()
    xyxy = []
    class_ids = []
    confidences = []

    for line in lines:
        class_id, x_center, y_center, width, height, confidence = line.split()
        x0 = float(x_center) - float(width) / 2
        y0 = float(y_center) - float(height) / 2
        x1 = float(x_center) + float(width) / 2
        y1 = float(y_center) + float(height) / 2
        x0 *= 640
        y0 *= 640
        x1 *= 640
        y1 *= 640
        xyxy.append([x0, y0, x1, y1])
        class_ids.append(class_id)
        confidences.append(confidence)

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confidences, dtype=float),
    )

    return detections


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
