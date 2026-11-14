import os
import cv2
import numpy as np
import pandas as pd
import concurrent.futures

BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
CSV_PATH = os.path.join(DATA_DIR, 'master_index_memmap.csv')  # original Training/Validation/Testing folders, same row order as memmap_idx
RAW_DIR = os.path.join(BASE_DIR, 'Comma_2k19_Driving_Dataset', 'Comma_2k19_Driving_Dataset')
FULLIMG_MEMMAP_PATH = os.path.join(DATA_DIR, 'fullimage_memmap.dat')

H_NEW, W_NEW = 144, 256
TOTAL_FRAMES = 91200

def get_raw_image(row):
    # set_id = "<Split>_Comma_2k19_<N>", filename = "<set_id>_image<N>_<frame>.jpg"
    split, folder = row['set_id'].split('_', 1)
    raw_name = row['filename'][len(row['set_id']) + 1:]
    raw_path = os.path.join(RAW_DIR, split, folder, raw_name)
    img = cv2.imread(raw_path)
    if img is None:
        raise FileNotFoundError(raw_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_row(idx_and_row):
    idx, row = idx_and_row
    img = get_raw_image(row)

    # 1. Resize to 256x192 preserving texture
    resized = cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA)
    
    # 2. Crop top 75% (Rows 0 through 143) to remove dashboard/hood
    cropped = resized[0:144, :, :]
    
    return row['memmap_idx'], cropped

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    # Pre-allocate memmap
    memmap = np.memmap(FULLIMG_MEMMAP_PATH, dtype='uint8', mode='w+',
                       shape=(TOTAL_FRAMES, H_NEW, W_NEW, 3))

    print(f"Resizing and cropping {len(df)} frames for full-image view...")

    # CPU-bound, so ProcessPool is ideal
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(process_row, df.iterrows(), chunksize=100)

        for i, (memmap_idx, processed_img) in enumerate(results):
            memmap[memmap_idx] = processed_img
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(df)} frames...")

    memmap.flush()
    print(f"Full-image precomputation complete. Saved to {FULLIMG_MEMMAP_PATH}")
