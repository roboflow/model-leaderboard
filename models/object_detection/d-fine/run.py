import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import supervision as sv
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from supervision.metrics import F1Score, MeanAveragePrecision
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

import multiprocessing

from configs import CONFIDENCE_THRESHOLD, DATASET_DIR
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    run_shell_command,
    write_result_json,
)

if not Path("D-FINE-repo").is_dir():
    run_shell_command(
        ["git", "clone", "https://github.com/Peterande/D-FINE.git", "./D-FINE-repo/"]
    )

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "./D-FINE-repo/"))
)

from src.core import YAMLConfig

LICENSE = "Apache-2.0"
RUN_PARAMETERS = dict(
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
    max_det=100,  # supervision uses internally, it is here just for logging
)
GIT_REPO_URL = "https://github.com/Peterande/D-FINE"
PAPER_URL = "https://arxiv.org/abs/2410.13842"

TRANSFORMS = T.Compose(
    [T.Resize((RUN_PARAMETERS["imgsz"], RUN_PARAMETERS["imgsz"])), T.ToTensor()]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_DICT = {
    "D-FINE-X-Objects365+COCO": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth",
        "model_filename": "dfine_x_obj2coco.pth",
        "model_name": "D-FINE-X-Objects365+COCO",
        "model_yaml": "./D-FINE-repo/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml",
    },
    "D-FINE-L-Objects365+COCO": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth",
        "model_filename": "dfine_l_obj2coco_e25.pth",
        "model_name": "D-FINE-L-Objects365+COCO",
        "model_yaml": "./D-FINE-repo/configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml",
    },
    "D-FINE-M-Objects365+COCO": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth",
        "model_filename": "dfine_m_obj2coco.pth",
        "model_name": "D-FINE-M-Objects365+COCO",
        "model_yaml": "./D-FINE-repo/configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.yml",
    },
    "D-FINE-S-Objects365+COCO": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth",
        "model_filename": "dfine_s_obj2coco.pth",
        "model_name": "D-FINE-S-Objects365+COCO",
        "model_yaml": "./D-FINE-repo/configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml",
    },
    "D-FINE-X": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth",
        "model_filename": "dfine_x_coco.pth",
        "model_name": "D-FINE-X",
        "model_yaml": "./D-FINE-repo/configs/dfine/dfine_hgnetv2_x_coco.yml",
    },
    "D-FINE-L": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth",
        "model_filename": "dfine_l_coco.pth",
        "model_name": "D-FINE-L",
        "model_yaml": "./D-FINE-repo/configs/dfine/dfine_hgnetv2_l_coco.yml",
    },
    "D-FINE-M": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
        "model_filename": "dfine_m_coco.pth",
        "model_name": "D-FINE-M",
        "model_yaml": "./D-FINE-repo/configs/dfine/dfine_hgnetv2_m_coco.yml",
    },
    "D-FINE-S": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "model_filename": "dfine_s_coco.pth",
        "model_name": "D-FINE-S",
        "model_yaml": "./D-FINE-repo/configs/dfine/dfine_hgnetv2_s_coco.yml",
    },
    "D-FINE-N": {
        "model_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth",
        "model_filename": "dfine_n_coco.pth",
        "model_name": "D-FINE-N",
        "model_yaml": "./D-FINE-repo/configs/dfine/dfine_hgnetv2_n_coco.yml",
    },
}  # noqa: E501 // docs


def download_weight(url, model_filename):
    run_shell_command(
        [
            "wget",
            "-O",
            model_filename,
            url,
        ]
    )


def run_on_image(model, image_array):
    im_pil = Image.fromarray(image_array[..., ::-1])
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(DEVICE)
    im_data = TRANSFORMS(im_pil).unsqueeze(0).to(DEVICE)
    output = model(im_data, orig_size)
    labels, boxes, scores = output
    class_id = labels.detach().cpu().numpy().astype(int)
    xyxy = boxes.detach().cpu().numpy()
    confidence = scores.detach().cpu().numpy()

    detections = sv.Detections(
        xyxy=xyxy[0],
        confidence=confidence[0],
        class_id=class_id[0],
    )
    detections = detections[detections.confidence > RUN_PARAMETERS.get("conf")]
    return detections


def evaluate_single_model(
    model_id: str, skip_if_result_exists: bool, dataset: Optional[sv.DetectionDataset]
):
    """
    Function to be run in a separate process for each model.
    """
    print(f"\nEvaluating model: {model_id}")
    model_values = MODEL_DICT[model_id]
    if skip_if_result_exists and result_json_already_exists(model_id):
        print(f"Skipping {model_id}. Result already exists!")
        return

    if dataset is None:
        dataset = load_detections_dataset(DATASET_DIR)

    if not os.path.exists(model_values["model_filename"]):
        download_weight(model_values["model_url"], model_values["model_filename"])

    # Re-initialize cfg and model for each iteration
    cfg = YAMLConfig(
        os.path.abspath(model_values["model_yaml"]),
        resume=model_values["model_filename"],
    )

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if model_values["model_filename"]:
        checkpoint = torch.load(model_values["model_filename"], map_location=DEVICE)
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model(cfg).to(DEVICE)

    predictions = []
    targets = []
    print(f"Evaluating {model_id}...")
    for _, image, target_detections in tqdm(dataset, total=len(dataset)):
        detections = run_on_image(model, image)
        predictions.append(detections)
        targets.append(target_detections)

    mAP_metric = MeanAveragePrecision()
    f1_score = F1Score()

    f1_score_result = f1_score.update(predictions, targets).compute()
    mAP_result = mAP_metric.update(predictions, targets).compute()

    write_result_json(
        model_id=model_id,
        model_name=model_values["model_name"],
        model_git_url=GIT_REPO_URL,
        paper_url=PAPER_URL,
        model=model,  # Consider if 'model' object needs to be passed, it might be large
        mAP_result=mAP_result,
        f1_score_result=f1_score_result,
        license_name=LICENSE,
        run_parameters=RUN_PARAMETERS,
    )
    print(f"mAP result 50:95 100 dets: {mAP_result.map50_95}")

    print(f"mAP result 50:95 100 dets rounded: {mAP_result.map50_95:.3f}")
    del model
    del cfg
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory freed for {model_id}.")
    print(f"Finished evaluating {model_id}. Cleaning up resources.")


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
        process = multiprocessing.Process(
            target=evaluate_single_model,
            args=(model_id, skip_if_result_exists, dataset),
        )
        process.start()
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    multiprocessing.set_start_method("fork", force=True)

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
