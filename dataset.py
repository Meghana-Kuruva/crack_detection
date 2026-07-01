import os
import yaml
from roboflow import Roboflow

# ==========================
# Roboflow Configuration
# ==========================
API_KEY = "Y7ZkdkpoRJdvjt4QsQod"
WORKSPACE = "meghana-x3ouv"
PROJECT = "twinsense-sleeper-crack-n0wvg-monii"
VERSION = 6
FORMAT = "yolov8-obb"

# ==========================
# Download Dataset
# ==========================
print("Connecting to Roboflow...")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

print("Downloading dataset...")
dataset = version.download(FORMAT)

dataset_dir = dataset.location
print(f"Dataset downloaded to: {dataset_dir}")

# ==========================
# Fix data.yaml
# ==========================
yaml_path = os.path.join(dataset_dir, "data.yaml")

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

# Set absolute dataset path
data["path"] = dataset_dir

with open(yaml_path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("\n✅ Dataset is ready!")
print(f"📂 Dataset Location: {dataset_dir}")
print(f"📄 data.yaml: {yaml_path}")

print("\nRun the pipeline using:")
print(f'python run_pipeline.py --data-yaml "{yaml_path}" --device 0')p