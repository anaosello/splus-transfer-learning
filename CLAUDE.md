# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an astronomy ML research project classifying S-PLUS (Southern Photometric Local Universe Survey) DR4 galaxy morphologies. The goal is to identify whether GalfitM morphological fitting results are "Good" (label 0 / `clase0`) or "Bad" (label 1 / `clase1`) — i.e., quality control of galaxy profile fitting outputs.

## Running Notebooks

All notebooks are designed to run on **Google Colab** with GPU. The data path is typically mounted from Google Drive:

```
/content/drive/MyDrive/Transfer learning/png_224
```

To run locally, update the path variables at the top of each notebook accordingly. Local runs require:

```bash
pip install torch torchvision transformers scikit-learn imbalanced-learn pandas numpy matplotlib astropy pillow
```

Run individual notebooks with:

```bash
jupyter notebook <notebook_name>.ipynb
# or
jupyter lab
```

## Notebook Architecture

### `example_pytorch_Transferlearning_CNN_vs_Transformer.ipynb`
Main experiment comparing two transfer learning approaches on galaxy images:
- **ViT (Vision Transformer)**: `Falconsai/nsfw_image_detection` from HuggingFace, backbone frozen, only classifier head fine-tuned. Uses `AutoImageProcessor` for preprocessing.
- **ResNet18 (CNN)**: ImageNet pretrained, backbone frozen, only the final `fc` layer replaced and fine-tuned.

Both models use **weighted cross-entropy loss** to handle class imbalance (~3731 Good vs ~552 Bad in training). Data augmentation: random horizontal/vertical flip + random 90° rotation.

The variants `3. (10)` and `3. (50)` are the same experiment run with 10 and 50 training epochs respectively.

### `random-forest.ipynb`
Tabular ML baseline using `BalancedRandomForestClassifier` on GalfitM morphological output features from `ALL_MorphoSPLUS_GalfitM_output_splus.csv`. Feature selection pipeline:
1. Univariate AUC filter (threshold > 0.60)
2. Pearson correlation deduplication (threshold 0.90, keep higher AUC feature)

Final selected features (~24): magnitude bands (`MAG_Z`, `MAG_J0515`, etc.), morphological parameters (`A`, `B`, `CHI2NU`, `FLUX_RADIUS_50/90`, `RE_J0660`, etc.).
Classification threshold is set to **0.25** (not 0.5) to favor recall on the minority "Bad" class.

### `analise_exploratoria_tabela copy.ipynb`
Exploratory data analysis of the tabular features: per-feature histograms, boxplots, and a Pearson correlation heatmap.

## Data Structure

```
png_224/
  train/
    clase0/   # Good fits (~3731 images)
    clase1/   # Bad fits (~552 images)
  val/
    clase0/
    clase1/
  test/
    clase0/
    clase1/

ALL_MorphoSPLUS_GalfitM_output_splus.csv   # Tabular morphological features (1442 rows, ~219 cols)
```

Image filenames follow the pattern:
`MorphoSPLUS_<field>_DR4_3_STRIPE82-<tile>_<object_id>[_<filter>].png`

Where `<filter>` is one of `g`, `r`, `z` (or absent for a composite/default image). Each galaxy object may have multiple images (one per filter).

## Key Design Decisions

- **Class imbalance**: ~87% Good / ~13% Bad. All models use weighted loss or balanced sampling. RF uses threshold=0.25 for predictions.
- **Transfer learning strategy**: Freeze entire backbone, only train classification head (linear layer). This avoids overfitting on the small minority class.
- **Metrics**: Balanced accuracy, per-class precision/recall/F1, and AUC are reported alongside standard accuracy, because accuracy alone is misleading given the imbalance.
- **Image preprocessing for ViT**: Uses the model's own `AutoImageProcessor` (not manual normalization), wrapped in a custom transform function to integrate with `torchvision.ImageFolder`.
