import argparse
import sys
from pathlib import Path
from typing import Optional

import supervision as sv
from inference import get_model
from supervision.metrics import F1Score, MeanAveragePrecision
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import CONFIDENCE_THRESHOLD, DATASET_DIR
from utils import (
    load_detections_dataset,
    result_json_already_exists,
    write_result_json,
)

MODEL_DICT = {
    "rfdetr-base": {"parameter_count": 29000000},
    "rfdetr-large": {"parameter_count": 128000000},
}
<<<<<<< HEAD

=======
CONFIDENCE_THRESHOLD = 0.5
>>>>>>> f797b7a0b1ce67d1575485f412dab51a0ddf4c2e
LICENSE = "APGL-3.0"
RUN_PARAMETERS = dict(
    conf=CONFIDENCE_THRESHOLD,
)
GIT_REPO_URL = "https://github.com/roboflow/rf-detr"
PAPER_URL = ""


def run_on_image(model, image) -> sv.Detections:
    detections = model.predict(image, confidence=CONFIDENCE_THRESHOLD)
    return detections


def run(
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
    for model_id in MODEL_DICT:
        if skip_if_result_exists and result_json_already_exists(model_id):
            print(f"Skipping {model_id}. Result already exists!")

        if dataset is None:
            dataset = load_detections_dataset(DATASET_DIR)

        model = get_model(model_id)

        predictions = []
        targets = []
        print("Evaluating...")
        for _, image, target_detections in tqdm(dataset, total=len(dataset)):
            # Run model
            detections = run_on_image(model, image)
            predictions.append(detections)
            targets.append(target_detections)

        mAP_metric = MeanAveragePrecision()
        f1_score = F1Score()

        f1_score_result = f1_score.update(predictions, targets).compute()
        mAP_result = mAP_metric.update(predictions, targets).compute()

        write_result_json(
            model_id=model_id,
            model_name=model_id,
            model_git_url=GIT_REPO_URL,
            paper_url=PAPER_URL,
            model=model,
            mAP_result=mAP_result,
            f1_score_result=f1_score_result,
            license_name=LICENSE,
            run_parameters=RUN_PARAMETERS,
            parameter_count=MODEL_DICT[model_id]["parameter_count"],
        )
        print(
            f"Finished evaluating {model_id}. Results saved. Map result: {mAP_result}, F1 score: {f1_score_result}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip_if_result_exists",
        action="store_true",
        help="If specified, skip the evaluation if the result json already exists.",
    )
    args = parser.parse_args()

    run(args.skip_if_result_exists)
