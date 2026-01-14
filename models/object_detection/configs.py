from pathlib import Path

CONFIDENCE_THRESHOLD = 0
PATH_TO_ROOT = Path(__file__).resolve().parent.parent.parent
PATH_TO_MODEL = PATH_TO_ROOT / "model"
PATH_TO_DATA = PATH_TO_ROOT / "data"
DATASET_DIR = str(PATH_TO_DATA  / "coco-val-2017")
