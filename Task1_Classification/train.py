import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils.zip_dataset import MultiTaskWikiArtZipDataset
from models.cnn_rnn import ArtCNNRNN

def count_classes(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return len(f.readlines())

def train_model(args):
    print(f"{'='*50}\nSTARTING ART-EXTRACT V2: WEIGHTED MULTI-TASK TRAINING\n{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device targeted: {device}")

    csv_dir = 'data/wikiart_csv'
    zip_path = 'data/wikiart.zip'

    num_artists = count_classes(os.path.join(csv_dir, 'artist_class.txt'))
    num_styles = count_classes(os.path.join(csv_dir, 'style_class.txt'))
    num_genres = count_classes(os.path.join(csv_dir, 'genre_class.txt'))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n--- Initializing Datasets ---")
    train_dataset = MultiTaskWikiArtZipDataset(zip_path, csv_dir, split='train', sample_fraction=args.sample, transform=transform)
    val_dataset = MultiTaskWikiArtZipDataset(zip_path, csv_dir, split='val', sample_fraction=args.sample, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ArtCNNRNN(num_artists, num_styles, num_genres).to(device)
    
    # --- V2 UPGRADE: Load EDA Class Weights ---
    print("\n--- Loading Inverse Class Weights ---")
    artist_w = torch.FloatTensor(np.load(os.path.join(csv_dir, 'artist_weights.npy'))).to(device)
    style_w = torch.FloatTensor(np.load(os.path.join(csv_dir, 'style_weights.npy'))).to(device)
    genre_w = torch.FloatTensor(np.load(os.path.join(csv_dir, 'genre_weights.npy'))).to(device)

    # Inject weights into the loss functions
    criterion_artist = nn.CrossEntropyLoss(weight=artist_w)
    criterion_style = nn.CrossEntropyLoss(weight=style_w)
    criterion_genre = nn.CrossEntropyLoss(weight=genre_w)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = 1

    history_train_loss = []
    history_val_loss = []

    # --- V2 UPGRADE: Gradient Accumulation (Simulating larger batch size) ---
    accumulation_steps = 4 # If batch_size=16, effective batch size = 64

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        
        # --- TRAIN PHASE ---
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad() # Zero gradients at start of epoch
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        for i, (images, labels) in train_bar:
            images = images.to(device)
            artist_tgt = labels['artist'].to(device)
            style_tgt = labels['style'].to(device)
            genre_tgt = labels['genre'].to(device)

            artist_pred, style_pred, genre_pred, _ = model(images)
            loss_artist = criterion_artist(artist_pred, artist_tgt)
            loss_style = criterion_style(style_pred, style_tgt)
            loss_genre = criterion_genre(genre_pred, genre_tgt)
            
            # Combine losses and divide by accumulation steps
            total_loss = (loss_artist + loss_style + loss_genre) / accumulation_steps
            total_loss.backward()
            
            # Step optimizer only after accumulating enough gradients
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            running_train_loss += total_loss.item() * accumulation_steps
            train_bar.set_postfix({'Loss': f"{total_loss.item() * accumulation_steps:.4f}"})
            
        avg_train_loss = running_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        
        val_bar = tqdm(val_loader, desc="Validating")
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                artist_tgt = labels['artist'].to(device)
                style_tgt = labels['style'].to(device)
                genre_tgt = labels['genre'].to(device)

                artist_pred, style_pred, genre_pred, _ = model(images)
                
                loss_artist = criterion_artist(artist_pred, artist_tgt)
                loss_style = criterion_style(style_pred, style_tgt)
                loss_genre = criterion_genre(genre_pred, genre_tgt)
                
                total_val_loss = loss_artist + loss_style + loss_genre
                running_val_loss += total_val_loss.item()
                val_bar.set_postfix({'Val Loss': f"{total_val_loss.item():.4f}"})

        avg_val_loss = running_val_loss / len(val_loader)
        
        print(f"-> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            save_path = 'checkpoints/best_cnn_rnn_v2.pth'
            torch.save(model.state_dict(), save_path)
            print(f"-> Target Acquired! V2 Model saved to {save_path}")

    # Generate Plot
    print("\nGenerating V2 training curve plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), history_train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, args.epochs + 1), history_val_loss, label='Val Loss', color='red', linewidth=2)
    plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.title('V2 Weighted Multi-Task CNN-RNN Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('checkpoints/loss_curve_v2.png')
    print("-> Plot saved successfully to checkpoints/loss_curve_v2.png")

    print("\nTraining V2 Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V2 CNN-RNN on WikiArt")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (will be accumulated x4)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sample', type=float, default=0.20, help='Fraction of data to use (0.0 to 1.0)')
    
    args = parser.parse_args()
    train_model(args)