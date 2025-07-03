import json
from pathlib import Path

# Temporarily disabled from appearing on the board, e.g. if there's still some issues
BLACKLIST = ["rtmdet", "yolov9"]

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

json_results = json.dumps(results_list, indent=2)
js_results = "const results = " + json_results + ";"

# Used to generate programmatic model comparison pages
with open("static/aggregate_results.json", "w") as f:
    f.write(json_results)

# Displayed in the table
with open("static/aggregate_results.js", "w") as f:
    f.write(js_results)

print("Results aggregated and saved to static/aggregate_results.js")
print("Results aggregated and saved to static/aggregate_results.json")
