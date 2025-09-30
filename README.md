# Histopathologic Cancer Detection (Kaggle Mini-Project)

## Project Overview

The task is to perform **binary classification** to detect metastatic cancer in small pathology image patches.

- **Dataset**: [Kaggle - Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)
- **Input**: 96×96 RGB `.tif` patches extracted from whole-slide images
- **Labels**: `0` = negative, `1` = positive (cancerous)
- **Evaluation Metric**: **ROC-AUC**

---

## Exploratory Data Analysis (EDA)

- Dataset is **slightly imbalanced** (~59% negative, ~41% positive).
- Sample patches show typical histopathologic color/texture variations.
- Channel statistics were computed, but for **consistency** with pretrained CNNs, **ImageNet normalization** was used.
- Visualization: class distribution plots, random sample grids, RGB histograms.

---

## Dataset & Augmentation

- **Augmentations**: horizontal/vertical flips, 90° rotations, color jitter.
- **Normalization**: ImageNet mean/std.
- **Batch size**: 128 (train), 256 (validation).
- **num_workers**: 2 (Colab GPU friendly).

---

## Model Architecture

- **Backbones tested**:
  - ResNet18 (baseline)
  - EfficientNet-B0 (main model)
- **Head**: single linear layer for binary classification.
- **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- **Scheduler**: CosineAnnealingLR
- **Loss**: BCEWithLogitsLoss

---

## Experiments

We systematically compared **ResNet18 vs EfficientNet-B0** under identical settings:

- **Epochs**: 5 (baseline comparison), 15 (final run)
- **Data fraction**: Due to Colab runtime, only **20% stratified sample** was used.
- **Metric**: Validation ROC-AUC

**Results (Validation AUC):**
| Model | Best AUC | Epoch |
|---------------|----------|-------|
| ResNet18 | ~0.978 | 5 |
| EfficientNet-B0 | ~0.988 | 5 |
| EfficientNet-B0 (final, 15 epochs) | ~0.990 | 10–13 |

EfficientNet-B0 consistently outperformed ResNet18.

---

## Training Curves

- **Loss**: steadily decreased, no strong overfitting observed.
- **ROC-AUC**: plateaued near **0.99** on validation.

---

## Inference & Submission

- Best model checkpoint (`effb0_best.pth`) was used.
- Predictions generated on test set → `submission.csv` for Kaggle upload.

**Leaderboard Scores:**

- **Public AUC**: 0.9621
- **Private AUC**: 0.9499

---

## Conclusion

### What worked

- **EfficientNet-B0** > ResNet18 under same budget (~0.99 ROC-AUC).
- **AdamW + cosine LR scheduler** yielded stable convergence.
- **Augmentations** (flips, rotations, jitter) improved generalization.

### Limitations

- Only **20% of dataset** was used due to Colab limits.
- Fixed **96×96 resolution**.
- No advanced preprocessing (e.g., stain normalization, WSI context).
- No ensembling or TTA applied.

### Future work

- Train on **full dataset** for leaderboard improvement.
- Apply **stain normalization / color constancy**.
- Use **TTA + ensembles** for boost.
- Explore **larger EfficientNets (B2/B3)** or modern backbones (ConvNeXt).
- Consider **semi-supervised pseudo-labeling**.

---

## References

- Kaggle Competition: [Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)
- Timm: PyTorch Image Models Library
- Albumentations: Image augmentation library
