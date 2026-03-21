from utils.zip_dataset import MultiTaskWikiArtZipDataset
from torchvision import transforms

def test_loader():
    # Simple transform to test the pipeline
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Initialize the dataset using 1% of the data just for a quick test
    dataset = MultiTaskWikiArtZipDataset(
        zip_path='data/wikiart.zip', 
        csv_dir='data/wikiart_csv', 
        split='train', 
        sample_fraction=0.01,  
        transform=test_transform
    )
    
    print("\n--- Testing Single Item Retrieval ---")
    image, labels = dataset[0]
    
    print(f"Image Tensor Shape: {image.shape}")
    print(f"Artist Label ID: {labels['artist']}")
    print(f"Style Label ID: {labels['style']}")
    print(f"Genre Label ID: {labels['genre']}")
    print("SUCCESS! Data pipeline is fully operational.")

if __name__ == "__main__":
    test_loader()