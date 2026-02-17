# SSHMT-MAE: Spatial-Spectral Hierarchical Multiscale Transformer for HSI Classification

## 📌 Project Overview
This repository contains my implementation and technical exploration of the **SSHMT-MAE** architecture, based on the 2025 research by Liu et al. (IEEE JSTARS). This project addresses the challenge of classifying Hyperspectral Images (HSI) using a self-supervised **Masked Autoencoder (MAE)** framework with a **Swin-based Hierarchical Transformer**.

By utilizing a 50% masking ratio and a novel window-grouping strategy, the model learns deep spatial-spectral representations from unlabeled data, significantly outperforming traditional CNN and ViT baselines.

## 🏗️ Technical Architecture & Math

### 1. Spatial-Spectral Feature Extraction (SSFE)
Traditional Patch Embedding often ignores the 3D structure of HSI. This implementation uses a 3D-CNN stem to preserve spectral signatures:
- **Operation:** Reduces spectral bands from $C \rightarrow 30$ via 2D Conv, followed by a $1 \times 1 \times 7$ 3D-CNN kernel.
- **Goal:** To capture the "Spectral Slope" (local band correlations) before entering the Transformer stages.

### 2. Grouped Window Attention (GWA)
Standard Swin Transformers process dummy "mask tokens," wasting GPU cycles. This project implements **GWA**:
- **Mechanism:** Uses an optimal grouping algorithm (based on the Knapsack problem) to pack only **visible tokens** into groups.
- **Math:** Reduces complexity from $O(L_{total}^2)$ to $O(L_{visible}^2)$, resulting in a **~25% reduction in pre-training latency**.

### 3. Hierarchical Multiscale Fusion (CFF)
The encoder consists of 4 stages with varying resolutions. The **Cross-Feature Fusion (CFF)** module integrates semantic high-level features with detailed low-level features using Global Average Pooling (GAP) and cross-stitching.

## 📊 Benchmarks (Indian Pines Dataset)
Based on the implementation and research benchmarks:
- **Overall Accuracy (OA):** 91.43%
- **Average Accuracy (AA):** 95.26%
- **Parameter Count:** 27.43M
- **Training Efficiency:** 0.94s / epoch (Pre-training)

| Method | Overall Accuracy (OA) | Gain |
| :--- | :--- | :--- |
| Standard ViT | 72.50% | - |
| 2D-CNN | 78.42% | - |
| **SSHMT-MAE (Ours)** | **91.43%** | **+18.93%** |

## 🛠️ Implementation Details
- **Framework:** PyTorch & TIMM (v0.4.12)
- **Masking Ratio:** 0.5 (Optimized for HSI spectral redundancy)
- **Loss Function:** MSE on masked pixels only.
- **Datasets Supported:** Indian Pines, Pavia University, Houston 2013.

## 🚀 How to Run
1. **Setup Environment:**
   ```bash
   pip install -r requirements.txt
