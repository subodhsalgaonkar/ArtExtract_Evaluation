import argparse
import os
import random
from utils.retrieve import PaintingRetriever
from utils.evaluation import ModelEvaluator

def run_ablation_study(query_path, top_k):
    print(f"\n{'='*60}")
    print(f"STARTING ABLATION STUDY FOR QUERY: {os.path.basename(query_path)}")
    print(f"{'='*60}\n")
    
    evaluator = ModelEvaluator()
    # We test ResNet first (Baseline), then DINOv2 (Proposed Method)
    models_to_test = ['resnet', 'dino']
    
    for model_type in models_to_test:
        print(f"\n--- Evaluating Model: {model_type.upper()} ---")
        retriever = PaintingRetriever(model_type=model_type)
        
        # Ask for top_k + 1 because the first result is usually the image itself
        results = retriever.find_similar(query_path, top_k=top_k + 1)
        
        print(f"\n[1] Generating Visual Grid for {model_type.upper()}...")
        retriever.visualize_results(query_path, results)
        
        print(f"\n[2] Running Comparative Evaluation Metrics for {model_type.upper()}...")
        evaluator.evaluate_retrieval(query_path, results)
        print("\n" + "-"*60)

def main():
    parser = argparse.ArgumentParser(description="ArtExtract Ablation Study Pipeline")
    parser.add_argument('--query', type=str, help="Path to query image. If none, picks a random gallery image.")
    parser.add_argument('--top_k', type=int, default=5, help="Number of similar images to retrieve")
    args = parser.parse_args()

    img_dir = 'data/images'
    query_path = args.query
    
    if not query_path:
        test_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        query_path = os.path.join(img_dir, random.choice(test_images))
    
    if not os.path.exists(query_path):
        print(f"Error: Query image {query_path} not found.")
        return

    run_ablation_study(query_path, args.top_k)

if __name__ == "__main__":
    main()