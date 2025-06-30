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

from configs import CONFIDENCE_THRESHOLD, DATASET_DIR
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    run_shell_command,
    write_result_json,
)

REPO_URL = "https://github.com/ShihuaHuang95/DEIM.git"
LICENSE = "Apache-2.0"
RUN_PARAMETERS = dict(
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
)
GIT_REPO_URL = "https://github.com/ShihuaHuang95/DEIM"
PAPER_URL = "https://arxiv.org/abs/2412.04234"
if not Path("./DEIM-repo/").is_dir():
    run_shell_command(["git", "clone", REPO_URL, "./DEIM-repo/"])

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "./DEIM-repo/"))
)
from engine.core import YAMLConfig


TRANSFORMS = T.Compose(
    [T.Resize((RUN_PARAMETERS["imgsz"], RUN_PARAMETERS["imgsz"])), T.ToTensor()]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DICT = {
    "DEIM-D-FINE-X": {
        "model_url": "https://drive.google.com/file/d/1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j/view?usp=drive_link",
        "model_filename": "dfine_x_coco.pth",
        "model_name": "DEIM-D-FINE-X",
        "model_yaml": "./DEIM-repo/configs/deim_dfine/deim_hgnetv2_x_coco.yml",
    },
    "DEIM-D-FINE-L": {
        "model_url": "https://drive.google.com/file/d/1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr/view?usp=drive_link",
        "model_filename": "dfine_l_coco.pth",
        "model_name": "DEIM-D-FINE-L",
        "model_yaml": "./DEIM-repo/configs/deim_dfine/deim_hgnetv2_l_coco.yml",
    },
    "DEIM-D-FINE-N": {
        "model_url": "https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC/view?usp=drive_link",
        "model_filename": "dfine_n_coco.pth",
        "model_name": "DEIM-D-FINE-N",
        "model_yaml": "./DEIM-repo/configs/deim_dfine/deim_hgnetv2_s_coco.yml",
    },
    "DEIM-D-FINE-S": {
        "model_url": "https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC/view?usp=drive_link",
        "model_filename": "dfine_s_coco.pth",
        "model_name": "DEIM-D-FINE-S",
        "model_yaml": "./DEIM-repo/configs/deim_dfine/deim_hgnetv2_s_coco.yml",
    },
    "DEIM-D-FINE-M": {
        "model_url": "https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8/view?usp=drive_link",
        "model_filename": "dfine_m_coco.pth",
        "model_name": "DEIM-D-FINE-M",
        "model_yaml": "./DEIM-repo/configs/deim_dfine/deim_hgnetv2_m_coco.yml",
    },
    "DEIM-RT-DETRv2-X": {
        "model_url": "https://drive.google.com/file/d/1BIevZijOcBO17llTyDX32F_pYppBfnzu/view?usp=drive_link",
        "model_filename": "deim_r101vd_60e_coco.pth",
        "model_name": "DEIM-RT-DETRv2-X",
        "model_yaml": "./DEIM-repo/configs/deim_rtdetrv2/deim_r101vd_60e_coco.yml",
    },
    "DEIM-RT-DETRv2-L": {
        "model_url": "https://drive.google.com/file/d/1mWknAXD5JYknUQ94WCEvPfXz13jcNOTI/view?usp=drive_link",
        "model_filename": "deim_r50vd_60e_coco.pth",
        "model_name": "DEIM-RT-DETRv2-L",
        "model_yaml": "./DEIM-repo/configs/deim_rtdetrv2/deim_r50vd_60e_coco.yml",
    },
    "DEIM-RT-DETRv2-S": {
        "model_url": "https://drive.google.com/file/d/153_JKff6EpFgiLKaqkJsoDcLal_0ux_F/view?usp=drive_link",
        "model_filename": "deim_r18vd_120e_coco.pth",
        "model_name": "DEIM-RT-DETRv2-S",
        "model_yaml": "./DEIM-repo/configs/deim_rtdetrv2/deim_r18vd_120e_coco.yml",
    },
    "DEIM-RT-DETRv2-M": {
        "model_url": "https://drive.google.com/file/d/1O9RjZF6kdFWGv1Etn1Toml4r-YfdMDMM/view?usp=drive_link",
        "model_filename": "deim_r34vd_120e_coco.pth",
        "model_name": "DEIM-RT-DETRv2-M",
        "model_yaml": "./DEIM-repo/configs/deim_rtdetrv2/deim_r34vd_120e_coco.yml",
    },
    "DEIM-RT-DETRv2-M*": {
        "model_url": "https://drive.google.com/file/d/10dLuqdBZ6H5ip9BbBiE6S7ZcmHkRbD0E/view?usp=drive_link",
        "model_filename": "deim_r50vd_m_60e_coco.pth",
        "model_name": "DEIM-RT-DETRv2-M*",
        "model_yaml": "./DEIM-repo/configs/deim_rtdetrv2/deim_r50vd_m_60e_coco.yml",
    },
}  # noqa: E501 // docs


def download_weight(url, model_filename):
    run_shell_command(
        [
            "gdown",
            f"{url.split('d/')[1].split('/view')[0]}",
            "-O",
            model_filename,
        ]
    )


def run_on_image(model, image_array):
    im_pil = Image.fromarray(image_array)
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
    return detections


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

        if not Path("DEIM-repo").is_dir():
            run_shell_command(["git", "clone", REPO_URL, "DEIM-repo"])

        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")
            continue

        if dataset is None:
            dataset = load_detections_dataset(DATASET_DIR)

        if not os.path.exists(model_values["model_filename"]):
            download_weight(model_values["model_url"], model_values["model_filename"])

        cfg = YAMLConfig(
            model_values["model_yaml"], resume=model_values["model_filename"]
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

        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        model = Model().to(DEVICE)

        predictions = []
        targets = []
        print("Evaluating...")
        for _, image, target_detections in tqdm(dataset, total=len(dataset)):
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
            model_name=model_values["model_name"],
            model_git_url=GIT_REPO_URL,
            paper_url=PAPER_URL,
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
