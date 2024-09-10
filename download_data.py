import os
from zipfile import ZipFile

import requests
from tqdm import tqdm

IMAGE_PATH = "data/coco-val-2017/images"
ANNOTATIONS_PATH = "data/coco-val-2017/labels"
# Create directories
os.makedirs(IMAGE_PATH, exist_ok=True)
os.makedirs(ANNOTATIONS_PATH, exist_ok=True)


# Function to download a file with progress bar
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(dest_path, "wb") as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong")


# Download and unzip test2017.zip
print("Downloading val2017.zip...")
test2017_zip_path = "data/val2017.zip"
download_file("http://images.cocodataset.org/zips/val2017.zip", test2017_zip_path)
print("Unzipping val2017.zip...")
with ZipFile(test2017_zip_path, "r") as zip_ref:
    zip_ref.extractall(IMAGE_PATH)
os.remove(test2017_zip_path)
print("val2017.zip downloaded and unzipped successfully.")

# Download and unzip image_info_test2017.zip
print("Downloading annotations_trainval2017.zip...")
image_info_zip_path = "data/annotations_trainval2017.zip"
download_file(
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    image_info_zip_path,
)
print("Unzipping annotations_trainval2017.zip...")
with ZipFile(image_info_zip_path, "r") as zip_ref:
    zip_ref.extractall(ANNOTATIONS_PATH)
os.remove(image_info_zip_path)
print("annotations_trainval2017.zip downloaded and unzipped successfully.")
