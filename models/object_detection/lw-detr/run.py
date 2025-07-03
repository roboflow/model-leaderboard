import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETRBase, RFDETRLarge
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

MODEL_DICT = {"RF-DETR-B": RFDETRBase, "RF-DETR-L": RFDETRLarge}
LICENSE = "Apache-2.0"
RUN_PARAMETERS = {
    "resolution": 560,
    "num_queries": 300,
    "num_select": 300,
    "threshold": 0,
}
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
            resolution=RUN_PARAMETERS["resolution"],
            num_queries=RUN_PARAMETERS["num_queries"],
            num_select=RUN_PARAMETERS["num_select"],
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
            detections.class_id = coco_id_vectorized_map(detections.class_id)
            predictions.append(detections)
            targets.append(target_detections)

        mAP_metric = MeanAveragePrecision()
        f1_metric = F1Score()
        f1_result = f1_metric.update(predictions, targets).compute()
        mAP_result = mAP_metric.update(predictions, targets).compute()

        write_result_json(
            model_id=model_id,
            model_name=model_id,
            model_git_url=GIT_REPO_URL,
            paper_url=PAPER_URL,
            model=model.model.model,
            mAP_result=mAP_result,
            f1_score_result=f1_result,
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


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    outputs = model(samples)

    results = postprocessors["bbox"](outputs, orig_target_sizes)
