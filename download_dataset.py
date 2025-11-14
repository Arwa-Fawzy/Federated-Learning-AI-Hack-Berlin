"""
Download pump sensor dataset from Kaggle
"""
import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("nphantawee/pump-sensor-data")

print("Path to dataset files:", path)

# Copy to current folder
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# Copy all files from downloaded path to data folder
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join(data_dir, file)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"Copied {file} to {data_dir}")

print(f"\nDataset files available in: {data_dir}")

