import pandas as pd
import os

def prepare_gallery_data(num_images=1000, output_csv='data/gallery_target.csv'):
    print("Loading NGA Metadata...")
    # Read the objects metadata
    objects_df = pd.read_csv('data/opendata/data/objects.csv', low_memory=False)
    
    # Read the image URLs metadata
    images_df = pd.read_csv('data/opendata/data/published_images.csv', low_memory=False)

    print("Filtering for Paintings...")
    # Filter only for Paintings
    paintings = objects_df[objects_df['classification'] == 'Painting']
    
    # Merge to get the IIIF image URLs (depictstmsobjectid maps to objectid)
    merged_df = paintings.merge(
        images_df, 
        left_on='objectid', 
        right_on='depictstmsobjectid', 
        how='inner'
    )

    # Drop items without a valid IIIF URL
    merged_df = merged_df.dropna(subset=['iiifurl'])

    # Sample a clean, randomized subset for our gallery (default 1000)
    # We use random_state so your results are reproducible
    gallery_subset = merged_df.sample(n=num_images, random_state=42).reset_index(drop=True)

    # Save the target data
    gallery_subset.to_csv(output_csv, index=False)
    print(f"Target gallery CSV created at {output_csv} with {len(gallery_subset)} paintings.")
    
    return gallery_subset

if __name__ == "__main__":
    prepare_gallery_data()
