from pathlib import Path
from typing import Any, Dict, List
import json

import gradio as gr


def load_results() -> List[Dict]:
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
    results_list.sort(key=lambda x: x["metadata"]["model"])
    return results_list

def get_result_header() -> List[str]:
    return ["Model", "Parameters (Mil.)", "mAP 50:95", "mAP 50", "mAP 75", "mAP 50:95 (Small)", "mAP 50:95 (Medium)", "mAP 50:95 (Large)"]

def parse_result(result: Dict) -> List[Any]:
    round_digits = 3

    param_count = ""
    if "param_count" in result["metadata"]:
        param_count = round(result["metadata"]["param_count"] / 1e6, 2)
    
    return [
        result["metadata"]["model"],
        param_count,
        round(result["map50_95"], round_digits),
        round(result["map50"], round_digits),
        round(result["map75"], round_digits),
        round(result["small_objects"]["map50_95"], round_digits),
        round(result["medium_objects"]["map50_95"], round_digits),
        round(result["large_objects"]["map50_95"], round_digits),
    ]


raw_results = load_results()
results = [parse_result(result) for result in raw_results]
header = get_result_header()

with gr.Blocks() as demo:
    gr.Markdown("# Model Leaderboard")
    gr.HTML("<italic>powered by: &nbsp<a href='https://github.com/roboflow/supervision'><img src='https://supervision.roboflow.com/latest/assets/supervision-lenny.png' height=24 width=24 style='display: inline-block'> supervision</a></italic>")
    gr.DataFrame(headers=header, value=results)

demo.launch()