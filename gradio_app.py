import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr

TITLE = """<h1 align="center">Model Leaderboard </h1>"""
DESC = """
<div style="text-align: center; display: flex; justify-content: center; align-items: center;">
    <italic>powered by: &nbsp<a href='https://github.com/roboflow/supervision'>
    <img src='https://supervision.roboflow.com/latest/assets/supervision-lenny.png'
    height=24 width=24 style='display: inline-block'> supervision</a></italic>
        <a href="https://github.com/roboflow/supervision">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/roboflow/supervision"
        style="margin-right: 10px;">
    </a>
</div>
"""  # noqa: E501 title/docs


def load_results() -> List[Dict]:
    results: List[Dict] = []
    results_file: Path = Path("static/aggregate_results.json")
    if not results_file.exists():
        print("aggregate_results.json file not found")
        sys.exit(1)
    with open(results_file) as f:
        results = json.load(f)
    results.sort(key=lambda x: x["metadata"]["model"])
    return results


def get_result_header() -> List[str]:
    return [
        "Model",
        "Parameters (M)",
        "mAP 50:95",
        "mAP 50",
        "mAP 75",
        "mAP 50:95 (Small)",
        "mAP 50:95 (Medium)",
        "mAP 50:95 (Large)",
        "F1 50",
        "F1 75",
        "F1 50 (Small)",
        "F1 75 (Small)",
        "F1 50 (Medium)",
        "F1 75 (Medium)",
        "F1 50 (Large)",
        "F1 75 (Large)",
    ]


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
        round(result["f1_50"], round_digits),
        round(result["f1_75"], round_digits),
        round(result["f1_small_objects"]["f1_50"], round_digits),
        round(result["f1_small_objects"]["f1_75"], round_digits),
        round(result["f1_medium_objects"]["f1_50"], round_digits),
        round(result["f1_medium_objects"]["f1_75"], round_digits),
        round(result["f1_large_objects"]["f1_50"], round_digits),
        round(result["f1_large_objects"]["f1_75"], round_digits),
    ]


raw_results = load_results()
results = [parse_result(result) for result in raw_results]
header = get_result_header()

with gr.Blocks() as demo:
    gr.Markdown("# Model Leaderboard")
    gr.HTML(TITLE)
    gr.HTML(DESC)
    gr.DataFrame(headers=header, value=results)

demo.launch()
