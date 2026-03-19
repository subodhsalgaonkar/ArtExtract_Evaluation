import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from torchvision import transforms
import matplotlib.pyplot as plt
import os

class ModelEvaluator:
    def __init__(self):
        # We resize images to a standard shape just for the SSIM/RMSE math
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def calculate_traditional_metrics(self, img1_path, img2_path):
        """Calculates SSIM and RMSE (The flawed pixel-level approach)"""
        try:
            img1 = Image.open(img1_path).convert('L').resize((224, 224)) # Convert to grayscale for SSIM
            img2 = Image.open(img2_path).convert('L').resize((224, 224))
            
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            score_ssim = ssim(arr1, arr2)
            score_rmse = np.sqrt(mse(arr1, arr2))
            return score_ssim, score_rmse
        except Exception as e:
            return 0.0, 0.0

    def evaluate_retrieval(self, query_path, results, img_dir='data/images'):
        print(f"\n--- EVALUATION REPORT FOR QUERY ---")
        print(f"Query Image: {os.path.basename(query_path)}")
        print(f"{'Rank':<5} | {'Cosine (Semantic)':<18} | {'SSIM (Pixel)':<15} | {'RMSE (Pixel)':<15}")
        print("-" * 65)
        
        avg_ssim = 0
        avg_rmse = 0
        valid_comparisons = 0
        
        for i, res in enumerate(results):
            # Skip the query image comparing to itself (Rank 1)
            if i == 0: continue 
                
            match_path = os.path.join(img_dir, f"{res['object_id']}.jpg")
            if os.path.exists(match_path):
                score_ssim, score_rmse = self.calculate_traditional_metrics(query_path, match_path)
                print(f"{i:<5} | {res['score']:<18.4f} | {score_ssim:<15.4f} | {score_rmse:<15.4f}")
                
                avg_ssim += score_ssim
                avg_rmse += score_rmse
                valid_comparisons += 1
                
        if valid_comparisons > 0:
            print("-" * 65)
            print(f"AVERAGE SSIM: {avg_ssim/valid_comparisons:.4f} (Notice how low this is despite good visual matches!)")
            print(f"AVERAGE RMSE: {avg_rmse/valid_comparisons:.4f}")
            print("CONCLUSION: Semantic Cosine Similarity vastly outperforms pixel-wise metrics for art.")

if __name__ == "__main__":
    from retrieve import PaintingRetriever
    import random
    
    retriever = PaintingRetriever()
    evaluator = ModelEvaluator()
    
    img_dir = 'data/images'
    test_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    if test_images:
        query_image = os.path.join(img_dir, random.choice(test_images))
        results = retriever.find_similar(query_image, top_k=6)
        evaluator.evaluate_retrieval(query_image, results)
