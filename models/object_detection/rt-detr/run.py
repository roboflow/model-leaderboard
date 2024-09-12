import argparse
import sys
from pathlib import Path
from typing import List, Optional

import supervision as sv
import torch
import torchvision.transforms as T
from configs import CONFIDENCE_THRESHOLD
from PIL import Image
from tqdm import tqdm
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
)

MODEL_DICT = {
    "rtdetr_r18vd": {"name": "RT-DETRv1 r18vd", "hub_id": "rtdetr_r18vd"},
    "rtdetr_r34vd": {"name": "RT-DETRv1 r34vd", "hub_id": "rtdetr_r34vd"},
    "rtdetr_r50vd": {"name": "RT-DETRv1 r50vd", "hub_id": "rtdetr_r50vd"},
    "rtdetr_r101vd": {"name": "RT-DETRv1 r101vd", "hub_id": "rtdetr_r101vd"},
    "rtdetrv2_r18vd": {"name": "RT-DETRv2 (r18vd)", "hub_id": "rtdetrv2_r18vd"},
    "rtdetrv2_r34vd": {"name": "RT-DETRv2 (r34vd)", "hub_id": "rtdetrv2_r34vd"},
    "rtdetrv2_r50vd": {"name": "RT-DETRv2 (r50vd)", "hub_id": "rtdetrv2_r50vd"},
    "rtdetrv2_r50vd_m": {"name": "RT-DETRv2 (r50vd_m)", "hub_id": "rtdetrv2_r50vd_m"},
    "rtdetrv2_r101vd": {"name": "RT-DETRv2 (r101vd)", "hub_id": "rtdetrv2_r101vd"},
}

sys.path.append(str(Path(__file__).resolve().parent.parent))

LICENSE = "Apache-2.0"
HUB_URL = "lyuwenyu/RT-DETR"
DATASET_DIR = "../../../data/coco-val-2017"
RUN_PARAMETERS = dict(
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
)

TRANSFORMS = T.Compose(
    [T.Resize((RUN_PARAMETERS["imgsz"], RUN_PARAMETERS["imgsz"])), T.ToTensor()]
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        model = torch.hub.load(HUB_URL, model_values["hub_id"], pretrained=True)
        model = model.to(DEVICE)

        predictions = []
        targets = []
        print("Evaluating...")
        for _, image, target_detections in tqdm(dataset, total=len(dataset)):
            img = Image.fromarray(image)
            width, height = img.size
            orig_size = torch.tensor([width, height])[None].to(DEVICE)
            im_data = TRANSFORMS(img)[None].to(DEVICE)

            results = model(im_data, orig_size)
            labels, boxes, scores = results
            class_id = labels.detach().cpu().numpy().astype(int)
            xyxy = boxes.detach().cpu().numpy()
            confidence = scores.detach().cpu().numpy()

            detections = sv.Detections(
                xyxy=xyxy[0],
                confidence=confidence[0],
                class_id=class_id[0],
            )

            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
            predictions.append(detections)

            target_detections.mask = None
            targets.append(target_detections)

        mAP_metric = sv.metrics.MeanAveragePrecision()
        mAP_result = mAP_metric.update(predictions, targets).compute()

        write_result_json(
            model_id=model_id,
            model_name=model_values["name"],
            model=model,
            mAP_result=mAP_result,
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
