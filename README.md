# S-PLUS Transfer Learning — Galaxy Morphology QC

Automated quality control of GalfitM morphological fits for S-PLUS DR4 galaxies.
Classifies each galaxy profile fit as **Good** (label 0) or **Bad** (label 1)
using transfer learning on galaxy images combined with tabular morphological features.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate splus-tl
```

---

## How to Predict on a New Dataset

The **Prediction on New Dataset** section at the bottom of `deit_finetune_ensemble.ipynb` is standalone —
you only need the saved weights, no retraining required.

**1. Make sure weights are in place:**

```
Transfer_learning/weights/
  vit-*.ckpt
  brf_tabular.pkl
  meta_lr.pkl
  features_sel.pkl
```

**2. Organize your images** using `ImageFolder` structure (at least one subfolder):

```
png_224_new/
  unknown/
    galaxy1.png
    galaxy2.png
    ...
```

> If your images are flat (no subfolders), create one:
> ```bash
> mkdir -p png_224_new/unknown && mv *.png png_224_new/unknown/
> ```

**3. Set the config variables** at the top of the prediction section:

```python
NEW_IMG_DIR = "png_224_new"          # folder with new images
NEW_TABLE   = "new_galaxies.csv"     # morphological features table (same columns as training)
NEW_ID_COL  = "ID_1"                 # ID column name in your table
OUTPUT_CSV  = "Transfer_learning/predictions_new.csv"
```

**4. Run all cells** in the prediction section.

The output CSV will contain:

| object_id | brf_prob | vit_prob | ensemble_prob | pred_label | pred_class |
|---|---|---|---|---|---|
| DR4_3_STRIPE82-0001_1234567 | 0.12 | 0.08 | 0.09 | 0 | Good |
| DR4_3_STRIPE82-0001_7654321 | 0.81 | 0.76 | 0.79 | 1 | Bad |

**5. If your new dataset has labels**, uncomment the evaluation block at the end
to get a classification report and confusion matrix.

---

## Notebooks

| Notebook | Model | Description |
|---|---|---|
| `deit_finetune_ensemble.ipynb` | DeiT-base (ViT) + BRF | Main model — transformer fine-tuning + stacking ensemble |
| `analise_exploratoria_tabela.ipynb` | — | EDA of morphological features |

---

## Running on Google Colab

Mount your Drive and update the path variables at the top of each section:

```python
from google.colab import drive
drive.mount("/content/drive")

pat         = "/content/drive/MyDrive/Transfer learning/png_224"
WEIGHTS_DIR = "/content/drive/MyDrive/Transfer learning/weights"
TABLE_PATH  = "/content/drive/MyDrive/Transfer learning/ALL_MorphoSPLUS_GalfitM_output_splus.csv"
LABELS_PATH = "/content/drive/MyDrive/Transfer learning/tabela_filtrada.csv"
```
