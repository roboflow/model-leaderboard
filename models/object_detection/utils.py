import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, List, Optional

import supervision as sv
from supervision.metrics import F1ScoreResult, MeanAveragePrecisionResult
from torch import nn


def get_images_directory_path(dataset_dir: str) -> str:
    return f"{dataset_dir}/images/val2017"


def get_annotations_path(dataset_dir: str) -> str:
    return f"{dataset_dir}/labels/annotations/instances_val2017.json"


def load_detections_dataset(dataset_dir: str) -> sv.DetectionDataset:
    print("Loading detections dataset...")
    return sv.DetectionDataset.from_coco(
        images_directory_path=get_images_directory_path(dataset_dir=dataset_dir),
        annotations_path=get_annotations_path(dataset_dir=dataset_dir),
    )


def download_file(url: str, output_filename: str) -> None:
    command = ["wget", url, "-O", output_filename]
    subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def run_shell_command(command: List[str], working_directory=None) -> None:
    subprocess.run(
        command,
        check=True,
        text=True,
        stdout=None,
        stderr=None,
        cwd=working_directory,
    )


def count_model_params(model: nn.Module) -> int:
    param_count = sum(p.numel() for p in model.parameters())
    return param_count


def _make_result_filename(model_id: str) -> str:
    return f"results_{model_id}.json"


def result_json_already_exists(model_id: str) -> bool:
    result_file = _make_result_filename(model_id)
    return os.path.exists(result_file)


def write_result_json(
    model_id: str,
    model_name: str,
    model_git_url: str,
    paper_url: Optional[str],
    model: 'nn.Module',
    mAP_result: Optional['MeanAveragePrecisionResult'] = None,
    f1_score_result: Optional['F1ScoreResult'] = None,
    license_name: str = "",
    run_parameters: dict[str, Any] = {},
    parameter_count: Optional[int] = None,
) -> None:
    result: dict[str, Any] = {}

    result["metadata"] = {
        "model": model_name,
        "license": license_name,
        "github_url": model_git_url,
        "paper_url": paper_url,
        "run_parameters": run_parameters,
        "param_count": count_model_params(model)
        if parameter_count is None
        else parameter_count,
        "run_date": datetime.now(timezone.utc).isoformat(),
    }

    # mAP metrics
    if mAP_result is not None:
        result["map50_95"] = mAP_result.map50_95
        result["map50"] = mAP_result.map50
        result["map75"] = mAP_result.map75

        result["small_objects"] = {
            "map50_95": mAP_result.small_objects.map50_95,
            "map50": mAP_result.small_objects.map50,
            "map75": mAP_result.small_objects.map75,
        }
        result["medium_objects"] = {
            "map50_95": mAP_result.medium_objects.map50_95,
            "map50": mAP_result.medium_objects.map50,
            "map75": mAP_result.medium_objects.map75,
        }
        result["large_objects"] = {
            "map50_95": mAP_result.large_objects.map50_95,
            "map50": mAP_result.large_objects.map50,
            "map75": mAP_result.large_objects.map75,
        }
        result["iou_thresholds"] = list(mAP_result.iou_thresholds)
    else:
        result["map50_95"] = None
        result["map50"] = None
        result["map75"] = None
        result["small_objects"] = {
            "map50_95": None,
            "map50": None,
            "map75": None,
        }
        result["medium_objects"] = {
            "map50_95": None,
            "map50": None,
            "map75": None,
        }
        result["large_objects"] = {
            "map50_95": None,
            "map50": None,
            "map75": None,
        }
        result["iou_thresholds"] = None

    # F1 metrics
    if f1_score_result is not None:
        result["f1_50"] = f1_score_result.f1_50
        result["f1_75"] = f1_score_result.f1_75

        result["f1_small_objects"] = {
            "f1_50": f1_score_result.small_objects.f1_50,
            "f1_75": f1_score_result.small_objects.f1_75,
        }
        result["f1_medium_objects"] = {
            "f1_50": f1_score_result.medium_objects.f1_50,
            "f1_75": f1_score_result.medium_objects.f1_75,
        }
        result["f1_large_objects"] = {
            "f1_50": f1_score_result.large_objects.f1_50,
            "f1_75": f1_score_result.large_objects.f1_75,
        }
        result["f1_iou_thresholds"] = list(f1_score_result.iou_thresholds)
    else:
        result["f1_50"] = None
        result["f1_75"] = None
        result["f1_small_objects"] = {
            "f1_50": None,
            "f1_75": None,
        }
        result["f1_medium_objects"] = {
            "f1_50": None,
            "f1_75": None,
        }
        result["f1_large_objects"] = {
            "f1_50": None,
            "f1_75": None,
        }
        result["f1_iou_thresholds"] = None

    result_file = _make_result_filename(model_id)
    if os.path.exists(result_file):
        os.rename(result_file, f"{result_file}.old")

    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)
