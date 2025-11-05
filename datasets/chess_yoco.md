---
task_categories:
- image-classification
task_ids:
- multi-class-image-classification
language:
- en
size_categories:
- 10K<n<100K
---

# Chess-Yoco Dataset

Dataset d'images d'échecs pour la classification des pièces.

## Structure

Le dataset contient des images de cases d'échiquier organisées en 13 classes :

- **Bishop_Black**, **Bishop_White** - Fous
- **King_Black**, **King_White** - Rois
- **Knight_Black**, **Knight_White** - Cavaliers
- **Pawn_Black**, **Pawn_White** - Pions
- **Queen_Black**, **Queen_White** - Dames
- **Rook_Black**, **Rook_White** - Tours
- **Empty** - Cases vides

## Splits

- **train/** : 33,997 images
- **validation/** : 4,245 images
- **test/** : 4,243 images

**Total** : ~42,485 images

## Utilisation

### Depuis Hugging Face

```python
from datasets import load_dataset

# Charger le dataset
dataset = load_dataset("nathbns/chess-yoco")

# Accéder aux splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

# Utiliser avec PyTorch
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
```

### Structure locale

```
chess-yoco-dataset/
├── train/
│   ├── Bishop_Black/
│   ├── Bishop_White/
│   ├── Empty/
│   └── ...
├── validation/
│   └── ...
└── test/
    └── ...
```

link:
https://huggingface.co/datasets/nathbns/chess-yoco