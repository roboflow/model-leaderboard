import argparse
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import supervision as sv
import torch
from PIL import Image
from rfdetr import (
    RFDETRBase,
    RFDETRLargeEdge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSmall,
    RFDETRXLCloud,
    RFDETRXXLCloud,
)
from rfdetr.util.coco_classes import COCO_CLASSES
from supervision.metrics import F1Score, MeanAveragePrecision
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import DATASET_DIR
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
)

ARCHITECTURE = "RF-DETR"
ARCHITECTURE_CHECKPOINTS = [
    "RF-DETR-B",
    "RF-DETR-N",
    "RF-DETR-S",
    "RF-DETR-M",
    "RF-DETR-L",
    "RF-DETR-XL",
    "RF-DETR-XXL",
]
MODEL_DICT = {
    "RF-DETR-B": RFDETRBase,
    "RF-DETR-N": RFDETRNano,
    "RF-DETR-S": RFDETRSmall,
    "RF-DETR-M": RFDETRMedium,
    "RF-DETR-L": RFDETRLargeEdge,
    "RF-DETR-XL": partial(RFDETRXLCloud, accept_platform_model_license=True),
    "RF-DETR-XXL": partial(RFDETRXXLCloud, accept_platform_model_license=True),
}
LICENSE = "Apache-2.0"
RUN_PARAMETERS = {
    # "resolution": 560,
    # "num_queries": 300,
    # "num_select": 300,
    "threshold": 0,
}
PRETRAIN_DATASETS = ["COCO", "Object365"]
GIT_REPO_URL = "https://github.com/roboflow/rf-detr"
PAPER_URL = ""


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_coco_id_mapping(coco_id_to_name, coco_classes_list):
    name_to_index = {name: idx for idx, name in enumerate(coco_classes_list)}
    coco_id_mapping = {}
    for coco_id, class_name in coco_id_to_name.items():
        if class_name in name_to_index:
            coco_id_mapping[coco_id] = name_to_index[class_name]
        else:
            continue
    return coco_id_mapping


def run(
    model_ids: List[str],
    skip_if_result_exists=False,
    dataset: Optional[sv.DetectionDataset] = None,
) -> None:
    if not model_ids:
        model_ids = list(MODEL_DICT.keys())

    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")

        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")
            continue

        if dataset is None:
            dataset = load_detections_dataset(DATASET_DIR)

        model = MODEL_DICT[model_id](
            device="cpu",
        )
        coco_id_mapping = create_coco_id_mapping(COCO_CLASSES, dataset.classes)
        coco_id_vectorized_map = np.vectorize(coco_id_mapping.__getitem__)

        predictions = []
        targets = []
        print("Evaluating...")
        for image_path, image, target_detections in tqdm(dataset, total=len(dataset)):
            image = Image.open(image_path).convert("RGB")
            detections = model.predict(image, threshold=RUN_PARAMETERS["threshold"])

            # workaround preventing disallowed class_ids
            allowed_class_id = list(coco_id_mapping.keys())
            detections = detections[np.isin(detections.class_id, allowed_class_id)]

            detections.class_id = coco_id_vectorized_map(detections.class_id)
            predictions.append(detections)
            targets.append(target_detections)

        mAP_metric = MeanAveragePrecision()
        f1_metric = F1Score()
        f1_result = f1_metric.update(predictions, targets).compute()
        mAP_result = mAP_metric.update(predictions, targets).compute()

        RUN_PARAMETERS.update(
            {
                "resolution": model.model_config.resolution,
                "num_queries": model.model_config.num_queries,
                "num_select": model.model_config.num_select,
            }
        )

        write_result_json(
            architecture=ARCHITECTURE,
            model_id=model_id,
            model_name=model_id,
            model_git_url=GIT_REPO_URL,
            paper_url=PAPER_URL,
            model=model.model.model,
            mAP_result=mAP_result,
            f1_score_result=f1_result,
            license=LICENSE if "X" not in model_id else "Commercial",
            run_parameters=RUN_PARAMETERS,
            pretrain_datasets=PRETRAIN_DATASETS,
            extra_metadata={"architecture_checkpoints": ARCHITECTURE_CHECKPOINTS},
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
