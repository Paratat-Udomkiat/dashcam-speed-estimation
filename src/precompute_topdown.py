import os
import cv2
import numpy as np
import pandas as pd
import concurrent.futures

BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
CSV_PATH = os.path.join(DATA_DIR, 'master_index_memmap.csv')  # original Training/Validation/Testing folders, same row order as memmap_idx
RAW_DIR = os.path.join(BASE_DIR, 'Comma_2k19_Driving_Dataset', 'Comma_2k19_Driving_Dataset')
TOPDOWN_MEMMAP_PATH = os.path.join(DATA_DIR, 'topdown_memmap.dat')

# Dimensions: H=192, W=96
H_NEW, W_NEW = 192, 96
TOTAL_FRAMES = 91200 # Update if your total frame count differs

# Warp at 4x width first, then downsample with INTER_AREA to avoid aliasing of the road texture
W_BIG = W_NEW * 4

# Compute Homography Matrix once
src_pts = np.array([[455, 480], [715, 480], [1164, 650], [0, 650]], dtype=np.float32)
dst_pts = np.array([[0, 0], [W_BIG, 0], [W_BIG, H_NEW], [0, H_NEW]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

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

    # Warp to Bird's Eye View
    warped = cv2.warpPerspective(img, M, (W_BIG, H_NEW), flags=cv2.INTER_LINEAR)
    warped = cv2.resize(warped, (W_NEW, H_NEW), interpolation=cv2.INTER_AREA)
    return row['memmap_idx'], warped

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    # Pre-allocate memmap
    memmap = np.memmap(TOPDOWN_MEMMAP_PATH, dtype='uint8', mode='w+',
                       shape=(TOTAL_FRAMES, H_NEW, W_NEW, 3))

    print(f"Warping {len(df)} frames to top-down view...")

    # CPU-bound, so ProcessPool is ideal
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(process_row, df.iterrows(), chunksize=100)

        for i, (memmap_idx, warped_img) in enumerate(results):
            memmap[memmap_idx] = warped_img
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(df)} frames...")

    memmap.flush()
    print(f"Top-down precomputation complete. Saved to {TOPDOWN_MEMMAP_PATH}")
