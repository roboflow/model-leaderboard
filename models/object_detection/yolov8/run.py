import sys
from pathlib import Path
from typing import List, Optional
import argparse

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_dataset_class_names, result_json_already_exists, write_result_json, load_detections_dataset


MODEL_IDS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
DATASET_DIR = "../../../data/coco-dataset"
CONFIDENCE_THRESHOLD = 0.001


def run_on_image(model, image) -> sv.Detections:
    result = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    map_class_ids_to_roboflow_format(detections)
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    return detections


def run(
    model_ids: List[str],
    skip_if_result_exists=False,
    dataset:Optional[sv.DetectionDataset] = None
) -> None:
    """
    Run the evaluation for the given models and dataset.
    
    Arguments:
        model_ids: List of model ids to evaluate. Evaluate all models if None.
        skip_if_result_exists: If True, skip the evaluation if the result json already exists.
        dataset: If provided, use this dataset for evaluation. Otherwise, load the dataset from the default directory.
    """
    if not model_ids:
        model_ids = MODEL_IDS

    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")

        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")
            continue

        if dataset is None: 
            dataset = load_detections_dataset(DATASET_DIR)

        model = YOLO(model_id)

        predictions = []
        targets = []
        print(f"Evaluating...")
        for _, image, target_detections in tqdm(dataset, total=len(dataset)):

            # Run model
            detections = run_on_image(model, image)
            predictions.append(detections)
            targets.append(target_detections)

        mAP_metric = sv.metrics.MeanAveragePrecision()
        mAP_result = mAP_metric.update(predictions, targets).compute()
        write_result_json(model_id, model, mAP_result)


def map_class_ids_to_roboflow_format(detections: sv.Detections) -> None:
    """
    Roboflow dataset has class names in alphabetical order. (airplane=0, apple=1, ...)
    Most other models use the COCO class order. (person=0, bicycle=1, ...).

    This function reads the class names from the detections, and remaps them to the Roboflow dataset order.
    """
    if "class_name" not in detections.data:
        raise ValueError("Detections should contain class names to reindex class ids.")

    dataset_class_names = get_dataset_class_names(DATASET_DIR)
    class_ids = [
        dataset_class_names.index(class_name)
        for class_name in detections.data["class_name"]
    ]
    detections.class_id = np.array(class_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_ids", nargs="*",
        help="Model ids to evaluate. If not provided, evaluate all models.",
    )
    parser.add_argument("--skip_if_result_exists", action="store_true",
        help="If specified, skip the evaluation if the result json already exists.",
    )
    args = parser.parse_args()

    run(args.model_ids, args.skip_if_result_exists)
