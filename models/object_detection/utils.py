import json
import subprocess
from datetime import datetime, timezone
from typing import Any, List

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecisionResult
from torch import nn


def download_file(url: str, output_filename: str) -> None:
    command = ["wget", url, "-O", output_filename]
    subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def count_model_params(model: nn.Module) -> int:
    param_count = sum(p.numel() for p in model.parameters())
    return param_count


def remap_class_ids(detections: sv.Detections, ordered_class_names: List[str]) -> None:
    """
    In-place remap the class_ids inside a Detections object, based on the class names.
    Useful when a model returns the correct class_names, but uses a different class_id indexing.
    """  # noqa: E501 // docs
    if "class_name" not in detections.data:
        raise ValueError("Detections should contain class names to reindex class ids.")

    class_ids = [
        ordered_class_names.index(class_name)
        for class_name in detections.data["class_name"]
    ]
    detections.class_id = np.array(class_ids)


def write_json_results(
    model_name: str, model: nn.Module, mAP_result: MeanAveragePrecisionResult
) -> None:
    result: dict[str, Any] = {}

    result["iou_thresholds"] = list(mAP_result.iou_thresholds)
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

    result["metadata"] = {
        "model": model_name,
        "param_count": count_model_params(model),
        "run_date": datetime.now(timezone.utc).isoformat(),
    }

    out_file = "results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=4)
