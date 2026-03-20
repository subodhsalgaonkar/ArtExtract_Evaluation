import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import os

class PaintingRetriever:
    def __init__(self, model_type='dino'):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        embeddings_path = f'data/gallery_embeddings_{model_type}.pt'
        print(f"Loading {model_type.upper()} gallery embeddings from {embeddings_path}...")
        data = torch.load(embeddings_path, map_location=self.device)
        self.gallery_embeddings = data['embeddings'].to(self.device)
        self.object_ids = data['object_ids']
        self.titles = data['titles']
        
        print(f"Loading {model_type.upper()} model for query extraction...")
        if model_type == 'dino':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        elif model_type == 'resnet':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.model.fc = nn.Identity()
            self.model = self.model.to(self.device)
            
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
            features = F.normalize(features, p=2, dim=1)
        return features

    def find_similar(self, query_path, top_k=5):
        query_emb = self.get_query_embedding(query_path)
        similarities = torch.mm(query_emb, self.gallery_embeddings.t()).squeeze(0)
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
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(20, 5))
        
        query_img = Image.open(query_path)
        axes[0].imshow(query_img)
        axes[0].set_title(f"QUERY IMAGE\n({self.model_type.upper()})", fontweight='bold')
        axes[0].axis('off')
        
        for i, res in enumerate(results):
            img_path = os.path.join(img_dir, f"{res['object_id']}.jpg")
            if os.path.exists(img_path):
                match_img = Image.open(img_path)
                axes[i+1].imshow(match_img)
                short_title = res['title'][:30] + '...' if len(res['title']) > 30 else res['title']
                axes[i+1].set_title(f"Match {i+1}\nScore: {res['score']:.3f}\n{short_title}", fontsize=9)
            axes[i+1].axis('off')
            
        plt.tight_layout()
        os.makedirs('doc', exist_ok=True)
        save_path = f'doc/similarity_visualization_{self.model_type}.png'
        plt.savefig(save_path)
        print(f"\nVisualization saved to {save_path}")
        plt.show()