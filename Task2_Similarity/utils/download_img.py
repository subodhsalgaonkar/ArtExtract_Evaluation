import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_image(row, save_dir='data/images'):
    object_id = row['objectid']
    base_url = row['iiifurl']
    
    # The IIIF API magic: we ask the server for a pre-resized 224x224 image
    # This saves massive amounts of bandwidth and local processing time
    image_url = f"{base_url}/full/!224,224/0/default.jpg"
    save_path = os.path.join(save_dir, f"{object_id}.jpg")
    
    # Skip if already downloaded
    if os.path.exists(save_path):
        return True
        
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        pass
    return False

def run_downloader(csv_path='data/gallery_target.csv', save_dir='data/images'):
    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting asynchronous download of {len(df)} images...")
    
    # We use ThreadPoolExecutor to download multiple images at the same time
    successful = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        # map() combined with tqdm gives us a nice progress bar
        results = list(tqdm(executor.map(lambda row: download_image(row[1], save_dir), df.iterrows()), total=len(df)))
    
    successful = sum(results)
    print(f"Download complete! Successfully fetched {successful}/{len(df)} images.")

if __name__ == "__main__":
    run_downloader()
