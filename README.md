# SiD-Diff: An Interpretable Siamese Distillation Framework for Detecting Differences in Hi-C Contact Maps

## Introduction
SiD-Diff is a deep learning framework designed to identify and interpret structural differences in 3D genome organization from Hi-C data. It addresses the lack of systematic characterization of differential chromatin interaction patterns and provides biologically meaningful insights into genome reorganization.

## Key Features
- **Data processing**：Construct paired samples from the datasets, using the K562-GM12878 pair as the training set. Segment the Hi-C contact maps along the diagonal using a sliding window approach with overlap compensation to ensure comprehensive coverage.Compute the differences between paired samples by integrating structural similarity and difference matrices. Samples with larger discrepancies are labeled as 1, while those with smaller discrepancies are labeled as 0. 
- **Manually selected data**：As illustrated, all segmented sample pairs are visualized for quality control. For samples labeled as 0, pairs that exhibit relatively larger differences are filtered out; similarly, for samples labeled as 1, pairs with relatively smaller differences are removed to improve label consistency and overall data quality.
- **Model architecture**：we developed SiD-Diff, which integrates convolutional neural networks (CNNs) and Transformers to model structural differences between paired Hi-C matrices, and incorporates biologically informed region encoding and cross-attention to enhance feature representation, together with knowledge distillation to improve model stability and generalization.
- **Interpretable Differential Patterns in 3D Genome Organization**: SiD-Diff identifies four key differential morphological patterns—boundary, stripe, asymmetric, and interior variations—through interpretability analysis. These patterns are strongly associated with dynamic changes in chromatin structural proteins such as CTCF and RAD21, highlighting the model’s ability to capture biologically meaningful features underlying 3D genome reorganization.
- **Cross-cell-line and cross-resolution universality**: SiD-Diff model demonstrates superior performance compared to Twins across both cross-cell-line and cross-resolution experiments.
- **Robust Detection of Chromatin Remodeling Across Conditions**: SiD-Diff effectively captures chromatin structural remodeling in both disease and perturbation settings. It accurately detects TAD alterations in breast cancer and structural changes following RAD21 knockout in HCT116 cells, demonstrating its robustness and interpretability for studying 3D genome reorganization.

## Steps to Install and Run SiD-Diff

### Clone the SiD-Diff repository
```bash
git clone https://github.com/Tong-Chen-ct/SiD-Diff.git
cd SiD-Diff
```

### Install the required dependencies
```
tensorflow>=2.4.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
biopython>=1.78
pybedtools>=0.8.0
matplotlib>=3.3.0
```

## Prepare your input data
### 1. Generate paired samples from two different .mcool files
Apply Strategy 6 for data segmentation to construct sample pairs and save them in .npz format.
```bash
python strategies_6.py
```
### 2.Visualize and curate the generated samples
Visualize the generated sample pairs and manually select high-quality samples to create a curated dataset.

## Train the CT-TADB model
Train the model and save the corresponding weights, including the Stage 1 student model weights, the Stage 2 teacher model weights, and the best-performing Stage 2 student model weights.
```bash
python SiD-Diff.py
```

## Model Architecture
- Siamese Framework: Takes paired Hi-C matrices as input and encodes them using shared-weight networks.
- CNN Feature Extraction: Two convolutional blocks (Conv: 16→32 channels, kernel sizes 5×5 and 3×3, with GELU, BatchNorm, MaxPooling, Dropout=0.3) capture local structural patterns.
- Biological Positional Encoding: Integrates genomic distance encoding (bin size = 5 kb, embed dim = 8) to model distance-dependent interaction decay.
- Transformer Module: A lightweight Transformer (embed dim = 16, heads = 2) models long-range dependencies via self-attention.
- Cross-Attention Mechanism: Bidirectional multi-head attention (heads = 2) enhances interaction between paired feature representations.
- Feature Fusion: Combines global pooled features and their absolute difference, followed by a dense layer (64 units) for representation learning.
- Classification Head: A sigmoid-activated output layer predicts similarity (0) or difference (1) between input pairs.
- Two-Stage Training with Distillation: Stage 1 uses BCE loss for pretraining; Stage 2 applies knowledge distillation with adaptive weighting to improve performance.

## File Structure
