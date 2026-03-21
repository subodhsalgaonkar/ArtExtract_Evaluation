import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.zip_dataset import MultiTaskWikiArtZipDataset
from models.cnn_rnn import ArtCNNRNN

def count_classes(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def evaluate_and_find_outliers():
    print(f"{'='*50}\nSTARTING T-SNE OUTLIER DETECTION\n{'='*50}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_dir = 'data/wikiart_csv'
    zip_path = 'data/wikiart.zip'
    
    artist_classes = count_classes(os.path.join(csv_dir, 'artist_class.txt'))
    style_classes = count_classes(os.path.join(csv_dir, 'style_class.txt'))
    genre_classes = count_classes(os.path.join(csv_dir, 'genre_class.txt'))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Validation Set (We only evaluate on data the model hasn't trained on)
    val_dataset = MultiTaskWikiArtZipDataset(zip_path, csv_dir, split='val', sample_fraction=0.20, transform=transform)
    # shuffle=False is CRITICAL here so we can match the embeddings back to the file paths
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Load Model
    model = ArtCNNRNN(len(artist_classes), len(style_classes), len(genre_classes)).to(device)
    model.load_state_dict(torch.load('checkpoints/best_cnn_rnn_v2.pth', map_location=device))
    model.eval()

    all_embeddings = []
    all_genres = []
    all_losses = []
    
    # We use loss to find the most "confusing" paintings
    criterion = nn.CrossEntropyLoss(reduction='none')

    print("\n--- Extracting High-Dimensional Embeddings ---")
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            genre_tgt = labels['genre'].to(device)
            
            _, _, genre_pred, embed = model(images)
            
            # Calculate loss per individual image
            loss = criterion(genre_pred, genre_tgt)
            
            all_embeddings.append(embed.cpu().numpy())
            all_genres.extend(genre_tgt.cpu().numpy())
            all_losses.extend(loss.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # --- FIND THE MATHEMATICAL OUTLIERS ---
    print("\n--- Identifying Top 3 Outliers ---")
    # Get the indices of the 3 highest loss values
    outlier_indices = np.argsort(all_losses)[-3:][::-1]
    
    for rank, idx in enumerate(outlier_indices):
        img_path = val_dataset.df.iloc[idx]['path']
        true_genre_idx = all_genres[idx]
        true_genre_name = genre_classes[true_genre_idx]
        loss_val = all_losses[idx]
        print(f"Outlier #{rank+1}: {img_path} | Labeled as: '{true_genre_name}' | Model Confusion (Loss): {loss_val:.4f}")

    # --- GENERATE T-SNE PLOT ---
    print("\n--- Running t-SNE Dimensionality Reduction (This may take a minute) ---")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_genres, cmap='tab10', alpha=0.7)
    
    # Highlight the top outlier on the graph
    top_outlier_idx = outlier_indices[0]
    plt.scatter(embeddings_2d[top_outlier_idx, 0], embeddings_2d[top_outlier_idx, 1], 
                color='red', edgecolor='black', s=200, marker='X', label='Top Outlier')

    plt.title('V2 t-SNE Projection of Weighted RNN "Art DNA"')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add a legend
    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [genre_classes[i] for i in range(len(handles))]
    plt.legend(handles, legend_labels, title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    os.makedirs('doc', exist_ok=True)
    plt.savefig('doc/tsne_outliers_v2.png', bbox_inches='tight')
    print("-> Scatter plot saved to doc/tsne_outliers_v2.png")

if __name__ == "__main__":
    evaluate_and_find_outliers()