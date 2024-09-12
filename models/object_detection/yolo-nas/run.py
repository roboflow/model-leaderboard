import argparse
import sys
from pathlib import Path
from typing import List, Optional

import super_gradients
import super_gradients.training
import super_gradients.training.models
import supervision as sv
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
)

LICENSE = "Apache-2.0"
MODEL_DICT = {
    "yolo_nas_s": {
        "name": "YOLO-NAS S",
    },
    "yolo_nas_m": {
        "name": "YOLO-NAS M",
    },
    "yolo_nas_l": {
        "name": "YOLO-NAS L",
    },
}
DATASET_DIR = "../../../data/coco-val-2017"
CONFIDENCE_THRESHOLD = 0.001


def run_on_image(model, image) -> sv.Detections:
    model_params = dict(
        iou=0.6,
        conf=0.001,
    )
    result = model.predict(image, **model_params)
    detections = sv.Detections.from_yolo_nas(result)
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
        model_ids = list(MODEL_DICT.keys())

    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")
        model_values = MODEL_DICT[model_id]

        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")
            continue

        if dataset is None:
            dataset = load_detections_dataset(DATASET_DIR)

        model = super_gradients.training.models.get(model_id, pretrained_weights="coco")
        if torch.cuda.is_available():
            model = model.cuda()

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
            model_name=model_values["name"],
            model=model,
            mAP_result=mAP_result,
            license_name=LICENSE,
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
        help="If specified, skip the evaluation if the result json already exists.",
        action="store_true",
    )
    args = parser.parse_args()

    run(args.model_ids, args.skip_if_result_exists)
