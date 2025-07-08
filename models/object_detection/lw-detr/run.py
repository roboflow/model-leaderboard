import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import supervision as sv
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image
from supervision.metrics import F1Score, MeanAveragePrecision
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import multiprocessing

from configs import CONFIDENCE_THRESHOLD, DATASET_DIR
from model_configs import (
    COCO_CLASSES,
    MODEL_CONFIGS,
    MODEL_DICT,
    REPO_ID,
    default_model_parameters,
)
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./LW-DETR/")))
from util.misc import nested_tensor_from_tensor_list

from models import build_model

LICENSE = "Apache-2.0"
HUB_URL = "lyuwenyu/RT-DETR"
RUN_PARAMETERS = dict(
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
)
GIT_REPO_URL = "https://github.com/lyuwenyu/RT-DETR"
PAPER_URL = "https://arxiv.org/abs/2304.08069"

TRANSFORMS = T.Compose(
    [T.Resize((RUN_PARAMETERS["imgsz"], RUN_PARAMETERS["imgsz"])), T.ToTensor()]
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_coco_id_mapping(coco_id_to_name, coco_classes_list):
    name_to_index = {name: idx for idx, name in enumerate(coco_classes_list)}
    coco_id_mapping = {}
    for coco_id, class_name in enumerate(coco_id_to_name):
        if class_name in name_to_index:
            coco_id_mapping[coco_id] = name_to_index[class_name]
        else:
            continue
    return coco_id_mapping


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    orig_image_size = torch.tensor(image.size[::-1])

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform = transforms.Compose(
        [
            transforms.Resize([640, 640]),
            normalize,
        ]
    )
    image = transform(image)
    return image, orig_image_size


def run_single_model(
    model_id: str,
    skip_if_result_exists=False,
    dataset: Optional[sv.DetectionDataset] = None,
) -> None:
    if skip_if_result_exists and result_json_already_exists(model_id):
        print(f"Skipping {model_id}. Result already exists!")
        return
    if dataset is None:
        dataset = load_detections_dataset(DATASET_DIR)
    local_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_DICT[model_id])

    model_cfg = MODEL_CONFIGS.get(model_id.lower(), None)
    if model_cfg is None:
        print(
            f"Skipping {model_id}. Model was not setup for running because is a Pre-Training checkpoint."  # noqa: E501
        )
        return
    model_cfg.update(default_model_parameters)
    cfg = SimpleNamespace(**model_cfg)

    model, criterion, postprocessors = build_model(cfg)
    checkpoint = torch.load(local_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    model.to(DEVICE)
    model.eval()
    criterion.eval()

    predictions = []
    targets = []
    print("Evaluating...")

    coco_id_mapping = create_coco_id_mapping(COCO_CLASSES, dataset.classes)
    coco_id_vectorized_map = np.vectorize(coco_id_mapping.__getitem__)

    for img_path, image, target_detections in tqdm(dataset, total=len(dataset)):
        image, orig_image_size = preprocess_image(img_path)
        image = image.to(DEVICE)
        orig_image_size = orig_image_size.to(DEVICE)
        images = nested_tensor_from_tensor_list([image])
        # forward
        with torch.no_grad():
            outputs = model(images)

        orig_image_sizes = torch.stack([orig_image_size])
        # postprocess
        preds = postprocessors["bbox"](outputs, orig_image_sizes)
        boxes = preds[0]["boxes"].cpu().numpy()
        labels = preds[0]["labels"].cpu().numpy()
        scores = preds[0]["scores"].cpu().numpy()
        class_id = np.atleast_1d(labels).astype(int)
        xyxy = np.atleast_2d(boxes)
        confidence = np.atleast_1d(scores)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]

        detections.class_id = coco_id_vectorized_map(detections.class_id)

        predictions.append(detections)

        target_detections.mask = None
        targets.append(target_detections)

    mAP_metric = MeanAveragePrecision()
    f1_metric = F1Score()

    f1_result = f1_metric.update(predictions, targets).compute()
    mAP_result = mAP_metric.update(predictions, targets).compute()

    model_name = (
        model_id.replace("_60e_coco", "").replace("_", "-").replace("LWDETR", "LW-DETR")  # noqa: E501
    )

    write_result_json(
        model_id=model_id,
        model_name=model_name,
        model_git_url=GIT_REPO_URL,
        paper_url=PAPER_URL,
        model=model,
        mAP_result=mAP_result,
        f1_score_result=f1_result,
        license_name=LICENSE,
        run_parameters=RUN_PARAMETERS,
    )
    print(f"mAP result 50:95 100 dets: {mAP_result.map50_95}")

    print(f"mAP result 50:95 100 dets rounded: {mAP_result.map50_95:.3f}")


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
        process = multiprocessing.Process(
            target=run_single_model, args=(model_id, skip_if_result_exists, dataset)
        )
        process.start()
        process.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip_if_result_exists",
        action="store_true",
        help="If specified, skip the evaluation if the result json already exists.",
    )
    args = parser.parse_args()

    run(MODEL_DICT.keys(), args.skip_if_result_exists)
