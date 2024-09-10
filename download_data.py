import roboflow

roboflow.login()
roboflow.download_dataset(
    dataset_url="https://universe.roboflow.com/microsoft/coco/dataset/3",
    model_format="yolov8",
    location="./data/coco-dataset",
)
