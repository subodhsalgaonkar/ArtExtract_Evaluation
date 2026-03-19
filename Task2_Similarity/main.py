import argparse
import os
import random
from utils.retrieve import PaintingRetriever
from utils.evaluation import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description="ArtExtract Similarity Search Pipeline")
    parser.add_argument('--query', type=str, help="Path to query image. If none, picks a random gallery image.")
    parser.add_argument('--top_k', type=int, default=5, help="Number of similar images to retrieve")
    args = parser.parse_args()

    # Initialize modules
    retriever = PaintingRetriever()
    evaluator = ModelEvaluator()
    img_dir = 'data/images'

    # Determine query image
    query_path = args.query
    if not query_path:
        test_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        query_path = os.path.join(img_dir, random.choice(test_images))
    
    if not os.path.exists(query_path):
        print(f"Error: Query image {query_path} not found.")
        return

    print(f"\n[1] Running Semantic Similarity Search for: {query_path}")
    # We ask for top_k + 1 because the first result is usually the image itself
    results = retriever.find_similar(query_path, top_k=args.top_k + 1)
    
    print("\n[2] Generating Visual Grid...")
    retriever.visualize_results(query_path, results)
    
    print("\n[3] Running Comparative Evaluation Metrics...")
    evaluator.evaluate_retrieval(query_path, results)

if __name__ == "__main__":
    main()
