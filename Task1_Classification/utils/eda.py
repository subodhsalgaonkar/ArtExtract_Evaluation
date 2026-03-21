import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def count_classes(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def analyze_distribution(csv_path, class_names, task_name):
    print(f"\n--- Analyzing {task_name.upper()} Distribution ---")
    
    # Read the training labels
    df = pd.read_csv(csv_path, header=None, names=['path', 'label_idx'])
    
    # Count frequencies
    counts = df['label_idx'].value_counts().sort_index()
    
    # Calculate Class Weights: Inverse Frequency
    # Formula: Total_Samples / (Number_of_Classes * Class_Frequency)
    total_samples = len(df)
    num_classes = len(class_names)
    
    weights = []
    print(f"Total Images: {total_samples} | Total Classes: {num_classes}")
    
    for i in range(num_classes):
        freq = counts.get(i, 0)
        if freq == 0:
            weight = 0.0
            print(f"WARNING: Class '{class_names[i]}' has 0 training samples!")
        else:
            weight = total_samples / (num_classes * freq)
        weights.append(weight)
        
    # Plotting the Imbalance
    plt.figure(figsize=(12, 8))
    
    # Sort for visual clarity (Long Tail)
    sorted_indices = np.argsort(counts.values)[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_counts = counts.values[sorted_indices]
    
    plt.bar(sorted_classes, sorted_counts, color='steelblue')
    plt.xticks(rotation=90)
    plt.title(f'WikiArt Class Imbalance: {task_name.capitalize()}')
    plt.ylabel('Number of Paintings')
    plt.tight_layout()
    
    os.makedirs('doc', exist_ok=True)
    plt.savefig(f'doc/eda_{task_name}_distribution.png')
    print(f"-> Distribution plot saved to doc/eda_{task_name}_distribution.png")
    
    return weights

if __name__ == "__main__":
    csv_dir = 'data/wikiart_csv'
    
    artists = count_classes(os.path.join(csv_dir, 'artist_class.txt'))
    styles = count_classes(os.path.join(csv_dir, 'style_class.txt'))
    genres = count_classes(os.path.join(csv_dir, 'genre_class.txt'))
    
    artist_weights = analyze_distribution(os.path.join(csv_dir, 'artist_train.csv'), artists, 'artist')
    style_weights = analyze_distribution(os.path.join(csv_dir, 'style_train.csv'), styles, 'style')
    genre_weights = analyze_distribution(os.path.join(csv_dir, 'genre_train.csv'), genres, 'genre')
    
    # Save weights to be loaded by the training script
    np.save('data/wikiart_csv/artist_weights.npy', artist_weights)
    np.save('data/wikiart_csv/style_weights.npy', style_weights)
    np.save('data/wikiart_csv/genre_weights.npy', genre_weights)
    print("\nSUCCESS: Inverse frequency weights saved. Ready for Model Preparation.")