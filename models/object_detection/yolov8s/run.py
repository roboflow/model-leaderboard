import sys
from pathlib import Path

import supervision as sv
import yaml
from torch import nn
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import remap_class_ids, write_json_results

MODEL_NAME = "yolov8s"
DATASET_DIR = "../../../data/coco-dataset"
CONFIDENCE_THRESHOLD = 0.001


def load_targets_and_make_predictions(model: nn.Module):
    with open(f"{DATASET_DIR}/data.yaml", "r") as f:
        dataset_yaml = yaml.safe_load(f)
        coco_class_names = dataset_yaml["names"]

    print("Loading test dataset...")
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{DATASET_DIR}/test/images",
        annotations_directory_path=f"{DATASET_DIR}/test/labels",
        data_yaml_path=f"{DATASET_DIR}/data.yaml",
    )

    predictions = []
    targets = []
    for _, image, target_detections in tqdm(dataset, total=len(dataset)):
        result = model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        remap_class_ids(detections, coco_class_names)
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        predictions.append(detections)

        target_detections.mask = None
        targets.append(target_detections)

    return predictions, targets


model = YOLO(f"{MODEL_NAME}.pt")

predictions, targets = load_targets_and_make_predictions(model)
mAP_metric = sv.metrics.MeanAveragePrecision()
mAP_result = mAP_metric.update(predictions, targets).compute()

write_json_results(MODEL_NAME, model, mAP_result)
