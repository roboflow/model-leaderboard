import sys
from typing import Dict
import numpy as np
import yaml
from pathlib import Path

import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import write_json_results


MODEL_NAME = "yolov9t"
DATASET_DIR = "../../../data/coco-dataset"
PREDICTIONS_DATASET_DIR = "yolov9/runs/detect/detection_out"
CONFIDENCE_THRESHOLD = 0.001


CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

with open(f"{DATASET_DIR}/data.yaml", "r") as f:
    dataset_yaml = yaml.safe_load(f)
    RF_CLASS_NAMES = dataset_yaml["names"]

CLASS_ID_TO_RF_CLASS_ID = {
    CLASS_NAMES.index(class_name): RF_CLASS_NAMES.index(class_name) for class_name in CLASS_NAMES
}

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
        confidence=np.array(confidences, dtype=float)
    )

    return detections

def load_predictions_dict(runs_dir: Path) -> Dict[str, sv.Detections]:
    print(f"Loading predictions dataset from {runs_dir}...")
    image_dir = runs_dir
    labels_dir = runs_dir / "labels"
    dataset = {}
    for image_path in image_dir.glob("*.jpg"):
        label_path = labels_dir / (image_path.stem + ".txt")
        detections = labels_to_detections(label_path)
        dataset[image_path.name] = detections
    return dataset

def load_targets_and_make_predictions():

    predictions_dict = load_predictions_dict(Path(PREDICTIONS_DATASET_DIR))

    print(f"Loading test dataset...")
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{DATASET_DIR}/test/images",
        annotations_directory_path=f"{DATASET_DIR}/test/labels",
        data_yaml_path=f"{DATASET_DIR}/data.yaml",
    )

    predictions = []
    targets = []
    for image_path, _, target_detections in tqdm(dataset, total=len(dataset)):
        detections = predictions_dict[Path(image_path).name]
        detections.class_id = np.array(
            [CLASS_ID_TO_RF_CLASS_ID[class_id] for class_id in detections.class_id])
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        predictions.append(detections)

        target_detections.mask = None
        targets.append(target_detections)

    return predictions, targets

model = YOLO(f"{MODEL_NAME}")

predictions, targets = load_targets_and_make_predictions()
mAP_metric = sv.metrics.MeanAveragePrecision()
mAP_result = mAP_metric.update(predictions, targets).compute()

write_json_results(MODEL_NAME, model, mAP_result)
