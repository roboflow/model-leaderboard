import argparse
import sys
from pathlib import Path
from typing import List, Optional

import supervision as sv
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm
from supervision.metrics import F1Score, MeanAveragePrecision
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import CONFIDENCE_THRESHOLD,DATASET_DIR
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    run_shell_command,
    write_result_json,
)


MODEL_DICT: dict = {
    "rtmdet_tiny_syncbn_fast_8xb32-300e_coco": {
        "model_name": "RTMDet-tiny",
        "config": "./mmyolo-weights/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py",
        "checkpoint_file": "./mmyolo-weights/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth",  # noqa: E501 // docs
    },
    "rtmdet_s_syncbn_fast_8xb32-300e_coco": {
        "model_name": "RTMDet-s",
        "config": "./mmyolo-weights/rtmdet_s_syncbn_fast_8xb32-300e_coco.py",
        "checkpoint_file": "./mmyolo-weights/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth",  # noqa: E501 // docs
    },
    "rtmdet_m_syncbn_fast_8xb32-300e_coco": {
        "model_name": "RTMDet-m",
        "config": "./mmyolo-weights/rtmdet_m_syncbn_fast_8xb32-300e_coco.py",
        "checkpoint_file": "./mmyolo-weights/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth",  # noqa: E501 // docs
    },
    "rtmdet_l_syncbn_fast_8xb32-300e_coco": {
        "model_name": "RTMDet-l",
        "config": "./mmyolo-weights/rtmdet_l_syncbn_fast_8xb32-300e_coco.py",
        "checkpoint_file": "./mmyolo-weights/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth",  # noqa: E501 // docs
    },
    "rtmdet_x_syncbn_fast_8xb32-300e_coco": {
        "model_name": "RTMDet-x",
        "config": "./mmyolo-weights/rtmdet_x_syncbn_fast_8xb32-300e_coco.py",
        "checkpoint_file": "./mmyolo-weights/rtmdet_x_syncbn_fast_8xb32-300e_coco_20221231_100345-b85cd476.pth",  # noqa: E501 // docs
    },
}

LICENSE = "GPL-3.0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_PARAMETERS = dict(
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
)


def run_on_image(model, image) -> sv.Detections:
    result = inference_detector(model, image)
    detections = sv.Detections.from_mmdetection(result)
    return detections


def download_weight(config_name):
    run_shell_command(
        [
            "mim",
            "download",
            "mmyolo",
            "--config",
            config_name,
            "--dest",
            "mmyolo-weights/",
        ]
    )


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

        download_weight(model_id)

        model = init_detector(
            model_values["config"], model_values["checkpoint_file"], DEVICE
        )

        predictions = []
        targets = []
        print("Evaluating...")
        for _, image, target_detections in tqdm(dataset, total=len(dataset)):
            # Run model
            detections = run_on_image(model, image)
            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
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
