import argparse
import sys
from pathlib import Path
from typing import List, Optional

import supervision as sv
from tqdm import tqdm
from ultralytics import YOLOv10

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
)

MODEL_IDS = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"]
DATASET_DIR = "../../../data/coco-val-2017"
CONFIDENCE_THRESHOLD = 0.001


def run_on_image(model, image) -> sv.Detections:
    result = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
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
        model_ids = MODEL_IDS

    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")

        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")
            continue

        if dataset is None:
            dataset = load_detections_dataset(DATASET_DIR)

        model = YOLOv10.from_pretrained(f"jameslahm/{model_id}")

        predictions = []
        targets = []
        print("Evaluating...")
        for _, image, target_detections in tqdm(dataset, total=len(dataset)):
            # Run model
            detections = run_on_image(model, image)
            predictions.append(detections)
            targets.append(target_detections)

        mAP_metric = sv.metrics.MeanAveragePrecision()
        mAP_result = mAP_metric.update(predictions, targets).compute()

        write_result_json(
            model_id=model_id,
            model_name=model_id,
            model=model,
            mAP_result=mAP_result
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
