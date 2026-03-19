import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

class PaintingRetriever:
    def __init__(self, embeddings_path='data/gallery_embeddings.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading gallery embeddings from {embeddings_path}...")
        data = torch.load(embeddings_path, map_location=self.device)
        self.gallery_embeddings = data['embeddings'].to(self.device)
        self.object_ids = data['object_ids']
        self.titles = data['titles']
        
        print("Loading DINOv2 model for query extraction...")
        # We load the exact same model to ensure the query is processed identically
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_query_embedding(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
            # CRITICAL: We must L2 normalize the query just like the gallery!
            features = F.normalize(features, p=2, dim=1)
        return features

    def find_similar(self, query_path, top_k=5):
        query_emb = self.get_query_embedding(query_path)
        
        # The Magic: Cosine Similarity via Dot Product
        similarities = torch.mm(query_emb, self.gallery_embeddings.t()).squeeze(0)
        
        # Fetch the indices of the Top K highest scores
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append({
                'score': score.item(),
                'object_id': self.object_ids[idx.item()],
                'title': self.titles[idx.item()]
            })
        return results

    def visualize_results(self, query_path, results, img_dir='data/images'):
        # Create a professional visual grid
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(20, 5))
        
        # Plot Query
        query_img = Image.open(query_path)
        axes[0].imshow(query_img)
        axes[0].set_title("QUERY IMAGE", fontweight='bold')
        axes[0].axis('off')
        
        # Plot Results
        for i, res in enumerate(results):
            img_path = os.path.join(img_dir, f"{res['object_id']}.jpg")
            if os.path.exists(img_path):
                match_img = Image.open(img_path)
                axes[i+1].imshow(match_img)
                # Display the similarity score and a truncated title
                short_title = res['title'][:30] + '...' if len(res['title']) > 30 else res['title']
                axes[i+1].set_title(f"Match {i+1}\nScore: {res['score']:.3f}\n{short_title}", fontsize=9)
            axes[i+1].axis('off')
            
        plt.tight_layout()
        # Save the visualization to the doc folder for your proposal
        os.makedirs('doc', exist_ok=True)
        plt.savefig('doc/similarity_visualization.png')
        print("\nVisualization saved to doc/similarity_visualization.png")
        plt.show()

if __name__ == "__main__":
    import random
    retriever = PaintingRetriever()
    
    # Grab a random image from our downloaded gallery to act as the query
    img_dir = 'data/images'
    test_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    if test_images:
        query_image = os.path.join(img_dir, random.choice(test_images))
        print(f"\nRunning test retrieval on: {query_image}")
        
        # We ask for the top 6 (The first result will be the image itself with a score of 1.0)
        results = retriever.find_similar(query_image, top_k=6)
        
        print("\nTop Matches Found:")
        for i, res in enumerate(results):
            print(f"{i+1}. Score: {res['score']:.4f} | Title: {res['title']}")
            
        retriever.visualize_results(query_image, results)
