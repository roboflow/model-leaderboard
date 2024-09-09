import sys
from pathlib import Path

import supervision as sv
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import remap_class_ids, write_json_results
from configs import COCO_CLASS_LIST,CONFIDENCE_THRESHOLD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "PekingU/rtdetr_r101vd"
DATASET_DIR = "../../../data/coco-dataset"


def load_targets_and_make_predictions(model: torch.nn.Module):
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
        img = Image.fromarray(image)
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        width, height = img.size
        target_size = torch.tensor([[height, width]])
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_size
        )[0]

        detections = sv.Detections.from_transformers(
            transformers_results=results, id2label=COCO_CLASS_LIST
        )

        remap_class_ids(detections, coco_class_names)
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        predictions.append(detections)

        target_detections.mask = None
        targets.append(target_detections)

    return predictions, targets


model = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE)
processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)


predictions, targets = load_targets_and_make_predictions(model)
mAP_metric = sv.metrics.MeanAveragePrecision()
mAP_result = mAP_metric.update(predictions, targets).compute()

write_json_results(MODEL_NAME, model, mAP_result)
