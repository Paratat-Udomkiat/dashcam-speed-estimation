import os
import cv2
import pandas as pd
import zipfile
from tqdm import tqdm

BASE = os.path.abspath('.')  # repository root (run scripts from here)
RAW_DATA_DIR = os.path.join(BASE, "Comma_2k19_Driving_Dataset", "Comma_2k19_Driving_Dataset")
OUTPUT_DIR = os.path.join(BASE, "processed_data")
ZIP_NAME = os.path.join(BASE, "dataset_128x384.zip")

os.makedirs(OUTPUT_DIR, exist_ok=True)
master_records = []

for split in ["Training", "Validation", "Testing"]:
    split_path = os.path.join(RAW_DATA_DIR, split)
    if not os.path.exists(split_path): continue

    for set_folder in os.listdir(split_path):
        set_path = os.path.join(split_path, set_folder)
        labels_csv = os.path.join(set_path, "labels.csv")

        if not os.path.exists(labels_csv): continue

        df = pd.read_csv(labels_csv)

        for _, row in df.iterrows():
            speed = row.get("car_speed_km_h")
            if pd.isna(speed) or speed == "":
                continue

            img_name = row["filename"]
            img_path = os.path.join(set_path, img_name)

            if not os.path.exists(img_path): continue

            img = cv2.imread(img_path)
            cropped = img[280:650, :]
            resized = cv2.resize(cropped, (384, 128), interpolation=cv2.INTER_AREA)

            new_filename = f"{split}_{set_folder}_{img_name}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, new_filename), resized)

            master_records.append({
                "split": split,
                "set_id": f"{split}_{set_folder}",
                "frame_id": int(row["frame"]), # Using true frame column
                "filename": new_filename,
                "speed": float(speed)
            })

master_df = pd.DataFrame(master_records)
master_df = master_df.sort_values(by=["set_id", "frame_id"]).reset_index(drop=True)
master_df.to_csv(os.path.join(OUTPUT_DIR, "master_index.csv"), index=False)

print("Packing uncompressed ZIP (ZIP_STORED)...")
with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_STORED) as zipf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in tqdm(files):
            zipf.write(os.path.join(root, file), file)
print(f"Done. Upload {ZIP_NAME} to Google Drive.")
