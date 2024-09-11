MODEL_DICT = {
    "rtdetrv2_r18vd": {
        "name": "RT-DETRv2 (r18vd)",
        "hub_id": "rtdetrv2_r18vd",
        "license": "Apache-2.0",
    },
    "rtdetrv2_r34vd": {
        "name": "RT-DETRv2 (r34vd)",
        "hub_id": "rtdetrv2_r34vd",
        "license": "Apache-2.0",
    },
    "rtdetrv2_r50vd": {
        "name": "RT-DETRv2 (r50vd)",
        "hub_id": "rtdetrv2_r50vd",
        "license": "Apache-2.0",
    },
    "rtdetrv2_r50vd_m": {
        "name": "RT-DETRv2 (r50vd_m)",
        "hub_id": "rtdetrv2_r50vd_m",
        "license": "Apache-2.0",
    },
    "rtdetrv2_r101vd": {
        "name": "RT-DETRv2 (r101vd)",
        "hub_id": "rtdetrv2_r101vd",
        "license": "Apache-2.0",
    },
}


model_ids = list(MODEL_DICT.keys())

for model_id in model_ids:
    print(f"\nEvaluating model: {model_id}")
    model_values = MODEL_DICT[model_id]
