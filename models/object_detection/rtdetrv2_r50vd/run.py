import sys
from pathlib import Path

import numpy as np
import supervision as sv
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from supervision.config import CLASS_NAME_DATA_FIELD
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import CLASS_NAMES, CONFIDENCE_THRESHOLD
from utils import remap_class_ids, write_json_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUB_URL = "lyuwenyu/RT-DETR"
MODEL_NAME = "rtdetrv2_r50vd"
DATASET_DIR = "../../../data/coco-dataset"
MODEL_FULL_NAME = "RT-DETRv2 (r50vd)"
TRANSFORMS = T.Compose([T.Resize((640, 640)), T.ToTensor()])


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
        width, height = img.size
        orig_size = torch.tensor([width, height])[None].to(DEVICE)
        im_data = TRANSFORMS(img)[None].to(DEVICE)

        results = model(im_data, orig_size)
        labels, boxes, scores = results
        class_id = labels.detach().cpu().numpy().astype(int)
        xyxy = boxes.detach().cpu().numpy()
        confidence = scores.detach().cpu().numpy()
        class_names = np.array([CLASS_NAMES[i] for i in class_id[0]])

        detections = sv.Detections(
            xyxy=xyxy[0],
            confidence=confidence[0],
            class_id=class_id[0],
            data={CLASS_NAME_DATA_FIELD: class_names},
        )
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]

        remap_class_ids(detections, coco_class_names)
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        predictions.append(detections)

        target_detections.mask = None
        targets.append(target_detections)

    return predictions, targets


model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)
model = model.to(DEVICE)

predictions, targets = load_targets_and_make_predictions(model)
mAP_metric = sv.metrics.MeanAveragePrecision()
mAP_result = mAP_metric.update(predictions, targets).compute()

write_json_results(MODEL_FULL_NAME, model, mAP_result)
