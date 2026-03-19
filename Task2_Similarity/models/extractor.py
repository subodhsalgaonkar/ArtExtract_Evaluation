import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --- 1. Dataset Class to load our downloaded Gallery ---
class GalleryDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Filter out any rows where the image might have failed to download
        self.valid_data = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, f"{row['objectid']}.jpg")
            if os.path.exists(img_path):
                self.valid_data.append((img_path, row['objectid'], row['title']))
                
    def __len__(self):
        return len(self.valid_data)
        
    def __getitem__(self, idx):
        img_path, obj_id, title = self.valid_data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, obj_id, title

# --- 2. The DINOv2 Feature Extractor ---
def extract_features(csv_file='data/gallery_target.csv', img_dir='data/images', output_file='data/gallery_embeddings.pt'):
    # A GPU (T4) is highly recommended for this step!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load DINOv2 (Vision Transformer Small - vits14)
    # This specific model balances lightning-fast speed with state-of-the-art accuracy
    print("Loading DINOv2 model from PyTorch Hub...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval() # Set to inference mode (no training needed!)
    
    # DINOv2 specific transforms (Resize to 224x224 and normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = GalleryDataset(csv_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Extracting 'Art DNA' (embeddings) for {len(dataset)} paintings...")
    
    all_embeddings = []
    all_ids = []
    all_titles = []
    
    with torch.no_grad():
        for images, obj_ids, titles in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            
            # Forward pass through DINOv2
            # It returns a tensor of shape [batch_size, 384]
            features = model(images) 
            
            # L2 Normalization: We normalize the vectors here so that calculating 
            # Cosine Similarity later is mathematically perfect and incredibly fast.
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            all_embeddings.append(features.cpu())
            all_ids.extend(obj_ids)
            all_titles.extend(titles)
            
    # Concatenate all batches into one massive tensor matrix
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save the embeddings dictionary to disk so we don't have to re-run this process
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save({
        'embeddings': final_embeddings,
        'object_ids': all_ids,
        'titles': all_titles
    }, output_file)
    
    print(f"\nSuccess! Saved {final_embeddings.shape[0]} embeddings to {output_file}.")

if __name__ == "__main__":
    extract_features()
