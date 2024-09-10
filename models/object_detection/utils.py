import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, List

import supervision as sv
import yaml
from supervision.metrics import MeanAveragePrecisionResult
from torch import nn


def load_detections_dataset(dataset_dir: str) -> sv.DetectionDataset:
    print("Loading detections dataset...")
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset_dir}/images/val2017",
        annotations_path=f"{dataset_dir}/labels/annotations/instances_val2017.json",
    )

    return dataset


def download_file(url: str, output_filename: str) -> None:
    command = ["wget", url, "-O", output_filename]
    subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def run_shell_command(command: List[str], working_directory=None) -> None:
    subprocess.run(
        command, check=True, text=True, stdout=None, stderr=None, cwd=working_directory
    )


def count_model_params(model: nn.Module) -> int:
    param_count = sum(p.numel() for p in model.parameters())
    return param_count


def _make_result_filename(model_name: str) -> str:
    return f"results_{model_name}.json"


def result_json_already_exists(model_name: str) -> bool:
    result_file = _make_result_filename(model_name)
    return os.path.exists(result_file)


def write_result_json(
    model_name: str, model: nn.Module, mAP_result: MeanAveragePrecisionResult
) -> None:
    result: dict[str, Any] = {}

    result["metadata"] = {
        "model": model_name,
        "param_count": count_model_params(model),
        "run_date": datetime.now(timezone.utc).isoformat(),
    }

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

    result_file = _make_result_filename(model_name)
    if os.path.exists(result_file):
        os.rename(result_file, f"{result_file}.old")

    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)
