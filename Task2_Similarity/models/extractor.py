import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse

class GalleryDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
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

def extract_features(model_type='dino', csv_file='data/gallery_target.csv', img_dir='data/images'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_file = f'data/gallery_embeddings_{model_type}.pt'
    
    # 1. Load the requested model
    if model_type == 'dino':
        print("Loading DINOv2 model...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif model_type == 'resnet':
        print("Loading ResNet50 (Optimized Base)...")
        # Using V2 weights which are significantly better than the original ImageNet1k weights
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Strip the final classification layer to get raw 2048-d features
        model.fc = nn.Identity() 
    else:
        raise ValueError("Invalid model type. Choose 'dino' or 'resnet'.")

    model = model.to(device)
    model.eval()
    
    # 2. Setup standard ImageNet transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = GalleryDataset(csv_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Extracting {model_type.upper()} embeddings for {len(dataset)} paintings...")
    
    all_embeddings = []
    all_ids = []
    all_titles = []
    
    with torch.no_grad():
        for images, obj_ids, titles in tqdm(dataloader, desc=f"Extracting ({model_type})"):
            images = images.to(device)
            features = model(images) 
            
            # L2 Normalization: Critical for BOTH models to ensure cosine similarity works mathematically
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            all_embeddings.append(features.cpu())
            all_ids.extend(obj_ids)
            all_titles.extend(titles)
            
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save({
        'embeddings': final_embeddings,
        'object_ids': all_ids,
        'titles': all_titles
    }, output_file)
    
    print(f"\nSuccess! Saved {final_embeddings.shape[0]} embeddings to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['dino', 'resnet'])
    args = parser.parse_args()
    extract_features(model_type=args.model)