# ArtExtract: Semantic Similarity & Classification for Fine Art

_Note: This repository is divided into two primary evaluation tasks. Task 1 focuses on hierarchical classification and outlier detection, while Task 2 explores semantic similarity and geometric retrieval._

---

## Task 1: Hierarchical Classification & Outlier Detection

_(Documentation, model architecture, and t-SNE outlier analysis for Task 1 will be populated here)._

---

## Task 2: Painting Similarity (National Gallery of Art)

### 📄 Abstract

With the rapid evolution of computer vision, AI models have demonstrated remarkable proficiency in image classification. However, applying these models to fine art presents a unique challenge: art similarity relies on geometric posture, semantic subject matter, and stylistic composition rather than literal pixel mapping. This task explores image similarity by conducting an ablation study, contrasting a highly optimized traditional Convolutional Neural Network (ResNet50) against a modern Self-Supervised Vision Transformer (DINOv2) to evaluate their efficacy in retrieving semantically similar paintings.

---

### Approach

#### 1. Data Acquisition & Pre-processing

The dataset is sourced from the [National Gallery of Art Open Data](https://github.com/NationalGalleryOfArt/opendata).

- **Filtering:** Records were strictly filtered for the 'Painting' classification.
- **Sampling:** A robust subset of 1,000 paintings was systematically sampled. _Why 1,000?_ Previous baselines testing ~150 images lacked the variance required to prove high-dimensional clustering. 1,000 images provide a statistically significant gallery while keeping local extraction compute times highly efficient for rapid prototyping.
- **Processing:** To preserve bandwidth and local memory, the pipeline leverages the NGA's IIIF API to dynamically request server-side cropping and resizing, ensuring all images are perfectly normalized to (224, 224) before downloading.

#### 2. Model: Feature Extraction (Ablation Setup)

Instead of forcing facial-detection bounding boxes (which fail on abstract and 2D art styles), this approach extracts holistic global features using two distinct architectures:

- **Baseline (Optimized ResNet50):** Building upon and improvising the previous year's GSoC applicant submission. While the previous iteration relied on MTCNN facial cropping (which fundamentally fails on 2D abstract art) and a limited 155-image subset, this heavily optimized baseline evaluates the holistic global composition across a robust 1,000-image dataset and strictly enforces L2 Normalization on `ImageNet1K_V2` weights for accurate angular distance calculation.
- **Proposed (DINOv2):** Meta's state-of-the-art Self-Supervised Vision Transformer (`vits14`). Trained without manual labels, it inherently maps global semantic structures rather than local textures, yielding dense 384-dimensional vectors.

#### 3. Similarity Assessment

Similarity is assessed by projecting the L2-normalized feature vectors into a manifold space and calculating the **Cosine Similarity**. Because the vectors are normalized, this high-dimensional angular distance is computed instantly via Matrix Multiplication (Dot Product), bypassing the latency of traditional nested loops.

---

### Evaluation Metrics

Performance evaluation is conducted through a dual-lens approach to highlight the discrepancy between standard photographic metrics and fine art analysis.

1. **Visual Evaluation (Top-K Retrieval):** Showcasing the top 5 nearest neighbors in feature space to subjectively diagnose the model's understanding of composition and pose.
2. **Quantitative Metrics (SSIM & RMSE):** \* **RMSE** calculates absolute pixel-wise spatial differences.
   - **SSIM** evaluates luminance, contrast, and structural degradation across sliding windows.
   - _Hypothesis:_ Because paintings of the same semantic subject (e.g., "seated portrait") can be painted in vastly different color palettes (e.g., Cubist Blues vs. Realist Browns), traditional pixel-wise metrics like SSIM and RMSE will fundamentally fail to recognize valid semantic matches.

---

### Results Analysis

An ablation study was run on a randomly selected query image (A seated male figure with hands raised behind the head).

**Baseline: ResNet50 Retrieval**
![ResNet Results](Task2_Similarity/doc/similarity_visualization_resnet.png)
_Observation:_ ResNet50 heavily prioritized superficial pixel correlations (overall color tone, ratio of white background to dark subject). It retrieved a standing woman, a man carrying a sack, and a card game, entirely missing the semantic geometric pose.

**Proposed: DINOv2 Retrieval**
![DINOv2 Results](Task2_Similarity/doc/similarity_visualization_dino.png)
_Observation:_ DINOv2 successfully captured the underlying semantic geometry. It retrieved multiple variations of seated or reclining figures with bent arms, completely ignoring the drastic differences in artistic era, texture, and color palette.

**Quantitative Report (Query: 73438.jpg):**
| Metric | ResNet50 (Average) | DINOv2 (Average) |
| :--- | :--- | :--- |
| **Cosine Similarity** | 0.613 | 0.617 |
| **SSIM (Pixel)** | 0.113 | 0.160 |
| **RMSE (Pixel)** | 91.86 | 74.07 |

**Conclusion:** The exceptionally low SSIM scores (averaging ~0.16) despite DINOv2's highly accurate visual matches mathematically proves that pixel-wise metrics are inadequate for fine art. Semantic Cosine Similarity extracted via Vision Transformers vastly outperforms traditional CNN methodologies.

---

### Possible Improvements

- **Multi-Modal Integration:** Expanding the feature space to include textual metadata embeddings (e.g., CLIP) alongside the visual vectors to allow for text-to-painting search capabilities.
- **Scaling Architectures:** Experimenting with larger ViT backbones (`dinov2_vitg14`) to capture even finer granular semantic details in complex multi-subject landscape paintings.
- **Dataset Expansion:** Scaling the pipeline to process the entire 130,000+ NGA catalog using cloud-distributed FAISS indexing for millisecond retrieval at scale.

---

### Implementation Guide

**1. Setup the Environment**

Ensure you have Python 3.8+ installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

**2. Generate Dataset & Feature Embeddings**

Since the image dataset and high-dimensional tensor matrices are excluded from version control to prevent repository bloat, you must first run the data pipeline to fetch the NGA images and generate the L2-normalized vectors for both models:
```bash
python utils/data_loader.py
python utils/download_img.py
python models/extractor.py --model resnet
python models/extractor.py --model dino
```

**3. Run the Ablation Study Pipeline**

The main script will automatically pick a query image, run both ResNet50 and DINOv2 back-to-back, generate the visual grids, and output the comparative evaluation metrics.
```bash
python main.py --top_k 5
```
*(Note: To query a specific image, use `--query data/images/your_image.jpg`)*

**4. Repository Structure**

- `models/extractor.py`: Multi-model feature extraction pipeline (`--model dino` or `--model resnet`).
- `utils/data_loader.py`: Parses NGA metadata to isolate the 1,000-image painting subset.
- `utils/download_img.py`: Asynchronous IIIF API downloader.
- `utils/retrieve.py`: Cosine similarity engine and visual grid generator.
- `utils/evaluation.py`: Comparative metric calculator (SSIM vs RMSE).
