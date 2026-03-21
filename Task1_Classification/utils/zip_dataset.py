import zipfile
import io
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MultiTaskWikiArtZipDataset(Dataset):
    def __init__(self, zip_path, csv_dir, split='train', sample_fraction=1.0, transform=None):
        """
        Reads images directly from a 27GB zip file while merging multi-task labels from CSVs.
        """
        self.zip_path = zip_path
        self.transform = transform
        self.zip_file = None # Opened lazily per-worker to prevent multiprocessing crashes
        
        # --- SMART DIRECTORY RESOLVER ---
        # Handles the common issue where extracting a zip creates a nested folder of the same name.
        actual_csv_dir = csv_dir
        if not os.path.exists(os.path.join(actual_csv_dir, f'artist_{split}.csv')):
            nested_dir = os.path.join(csv_dir, 'wikiart_csv')
            if os.path.exists(os.path.join(nested_dir, f'artist_{split}.csv')):
                print(f"[{split.upper()} DATASET] Detected nested CSV folder structure. Adjusting path...")
                actual_csv_dir = nested_dir
            else:
                raise FileNotFoundError(f"Could not find artist_{split}.csv in {csv_dir}. Please check your extraction.")
        
        print(f"\n[{split.upper()} DATASET] Loading CSVs from {actual_csv_dir}...")
        
        # 1. Read the three separate CSV files using the resolved directory
        df_artist = pd.read_csv(os.path.join(actual_csv_dir, f'artist_{split}.csv'), header=None, names=['path', 'artist_idx'])
        df_style  = pd.read_csv(os.path.join(actual_csv_dir, f'style_{split}.csv'), header=None, names=['path', 'style_idx'])
        df_genre  = pd.read_csv(os.path.join(actual_csv_dir, f'genre_{split}.csv'), header=None, names=['path', 'genre_idx'])
        
        # 2. Merge them into a single Master DataFrame based on the image path
        print(f"[{split.upper()} DATASET] Merging labels for multi-task training...")
        df_merged = df_artist.merge(df_style, on='path', how='inner').merge(df_genre, on='path', how='inner')
        
        # 3. Sample the data for rapid local testing
        if sample_fraction < 1.0:
            self.df = df_merged.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
            print(f"[{split.upper()} DATASET] Subsampled to {len(self.df)} images ({sample_fraction*100}%).")
        else:
            self.df = df_merged
            print(f"[{split.upper()} DATASET] Loaded all {len(self.df)} images.")

        # 4. Smart Path Resolution for the Zip File
        print(f"[{split.upper()} DATASET] Scanning zip archive structure...")
        with zipfile.ZipFile(self.zip_path, 'r') as archive:
            all_zip_paths = archive.namelist()
            
        sample_path = self.df.iloc[0]['path'].replace('\\', '/')
        self.zip_prefix = ""
        for zp in all_zip_paths:
            if zp.endswith(sample_path):
                self.zip_prefix = zp.replace(sample_path, '')
                break
        print(f"[{split.upper()} DATASET] Zip internal prefix detected as: '{self.zip_prefix}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # We only open the zip file connection when __getitem__ is actually called
        if self.zip_file is None:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
            
        row = self.df.iloc[idx]
        
        # Ensure correct slash direction for reading inside zip files
        rel_path = row['path'].replace('\\', '/')
        full_zip_path = self.zip_prefix + rel_path
        
        # Extract the image directly into RAM as bytes
        image_bytes = self.zip_file.read(full_zip_path)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return the image and a dictionary of its three multi-task labels
        labels = {
            'artist': torch.tensor(row['artist_idx'], dtype=torch.long),
            'style':  torch.tensor(row['style_idx'], dtype=torch.long),
            'genre':  torch.tensor(row['genre_idx'], dtype=torch.long)
        }
            
        return image, labels