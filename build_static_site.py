import json
from pathlib import Path

# Temporarily disabled from appearing on the board, e.g. if there's still some issues
BLACKLIST = ["yolov9", "yolo-nas"]

results_list = []
for model_dir in Path("models/object_detection").iterdir():
    if model_dir.is_file() or model_dir.name.startswith("_"):
        continue

    if model_dir.name in BLACKLIST:
        continue

    for results_file in model_dir.glob("results*.json"):
        with open(results_file) as f:
            results = json.load(f)
            results_list.append(results)

aggregate_results = "const results = " + json.dumps(results_list, indent=2) + ";"
with open("static/aggregate_results.js", "w") as f:
    f.write(aggregate_results)

print("Results aggregated and saved to static/aggregate_results.js")
