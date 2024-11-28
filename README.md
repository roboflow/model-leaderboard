# Model Leaderboard Website

This project is a **model leaderboard** website for evaluating a variety of models on the COCO dataset. Models are compared compares models across a variety of metrics, computed with [supervision](https://github.com/roboflow/supervision).

## Project Structure

```text
model-leaderboard/
│
├── data/                  # Directory for storing datasets (e.g., COCO)
├── models/                # Model-specific directories
│   ├── object_detection/   # Houses individual model folders
│   │   ├── yolov8n/        # Example of a model folder
│   │   ├── yolov9t/        # Example of another model folder
│   │   └── yolov10m/       # And so on...
│   └── utils.py            # Shared utility code (minimal cross-model dependencies)
│
├── index.html              # Main page for the static side
├── static/                 # Static files, serving as the backend for the site
│   └── ...
├── download_data.py        # Script for downloading the ground truth dataset
├── build_static_site.py    # Must be run to aggregate results for the static site
└── requirements.txt        # Dependencies for data download and leaderboard front end
```

Each model is expected to have:

```text
model_name/
│
├── requirements.txt        # Dependencies for running the model
├── run.py                  # Script that runs model on dataset in /data, compares with ground truth
├── results.json            # Model results, created with run.py
├── (any other scripts)
└── (optional) README.md    # Links to original model page & instructions on how to produce results.py
```

Each model is expected to be run in a separate python virtual environment, with its own set of dependencies.

Before we automate the models to be run regularly, the naming standards are relaxed.
The only requirements is to store results for `results.json` in the model directory. For consistency, we advice to keep the scripts in `run.py`.

### Key Files

1. **`download_data.py`**: Downloads the dataset (currently configured for COCO) and places it into the `data/` directory.
2. **`build_static_site.py`**: Aggregates the results of the models to be shown on the GitHub Pages site.
3. **`run_overnight.sh`**: An early version of a script to run the entire process, generating model results and comparing to downloaded data. We hacked it together for the first iteration of the leaderboard. Requires `uv`.
4. **`gradio_app.py`**: The initial version of the leaderboard UI. Displays model results in a gradio page.

5. **Model-Specific Folders**"

   - Each object detection model is housed in its own folder under `models/object_detection/`. These folders include `run.py`, which generates evaluation results in the form of a `results.json` file.

6. **`utils.py`**: Contains shared utility code across models.

## Getting Started

### 1. Download the Dataset

Run the `download_data.py` script to download the COCO dataset (or any supported dataset in the future):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
deactivate
```

### 2. Run the model

Each model folder contains its own script to run predictions and generate results.

```bash
cd models/object_detection/yolov8n
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py  # Or any other scripts
deactivate
```

### 3. Run All Models

Currently, you may use `run_overnight.sh` to run all models.
This is an early version of the script and is due to change.

```bash
./run_overnight.sh
```

### 4. Launch the Leaderboard

Once the results are generated, you can launch the Gradio app to visualize the leaderboard:

```bash
source .venv/bin/activate
python gradio_app.py
```

## Notes

- The leaderboard is currently configured to work with the COCO dataset, running the benchmark on the test set.
- Model dependencies are handled separately in each model folder, with each model having its own `requirements.txt` to avoid dependency conflicts.

## Contributing

Feel free to contribute by adding new models, improving the existing evaluation process, or expanding dataset support. To add a new model, simply create a new folder under `models/object_detection/` and follow the structure of existing models.

Make sure that the model's `run.py` script generates a `results.json` file in the model directory. If there are more scripts, please provide a `README.md` in the model folder how to run the model. Optionally, you may modify the `run_overnight.sh` script to help us automate the benchmarking in the future.
