import json
from pathlib import Path

results_list = []
for model_dir in Path("models/object_detection").iterdir():
    if model_dir.is_file() or model_dir.name.startswith("_"):
        continue

    results_file = model_dir / "results.json"
    if not results_file.exists():
        print(f"Results file not found for {model_dir.name}")
        continue

    with open(results_file) as f:
        results = json.load(f)
        results_list.append(results)

aggregate_results = "const results = " + json.dumps(results_list, indent=2) + ";"
with open("static/aggregate_results.js", "w") as f:
    f.write(aggregate_results)

print("Results aggregated and saved to static/aggregate_results.js")
