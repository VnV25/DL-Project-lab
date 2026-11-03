"""Dataset and image transforms.

The training augmentations imitate how real people photograph notes, which is what makes
the model work on photos it has never seen:
  * any orientation (rotated 90/180/270 degrees, slightly tilted)
  * note only partly in the frame, or far away        -> RandomResizedCrop
  * photo taken at an angle                            -> RandomPerspective
  * dim room, bright sunlight, different phone colours -> ColorJitter (hue kept tiny,
                                                          because colour tells notes apart)
  * shaky hands, out of focus                          -> GaussianBlur
  * fingers covering part of the note                  -> RandomErasing
"""
import csv
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

import config


class RandomRot90:
    """Rotate by 0, 90, 180 or 270 degrees (notes can be held any way up)."""

    def __call__(self, img):
        k = random.randint(0, 3)
        ops = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]
        return img if k == 0 else img.transpose(ops[k])


def train_transforms(size=config.IMG_SIZE):
    return T.Compose([
        RandomRot90(),
        T.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.5, 2.0)),
        T.RandomApply([T.RandomRotation(15)], p=0.5),
        T.RandomPerspective(distortion_scale=0.25, p=0.4),
        T.ColorJitter(brightness=0.45, contrast=0.4, saturation=0.35, hue=0.02),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25),
        T.ToTensor(),
        T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


def eval_transforms(size=config.IMG_SIZE):
    # Whole image squeezed to a square: the same thing RandomResizedCrop does at full scale,
    # and it never cuts off the printed value at the edge of a long note.
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def read_splits():
    with open(config.SPLITS_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class NoteDataset(Dataset):
    def __init__(self, rows, classes, transform):
        self.rows = rows
        self.index = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(config.ROOT / r["path"]).convert("RGB")
        return self.transform(img), self.index[r["label"]]


def class_weights(rows, classes):
    """Give rarer classes more weight so the model doesn't ignore them."""
    counts = torch.tensor([sum(r["label"] == c for r in rows) for c in classes], dtype=torch.float)
    w = counts.sum() / (len(classes) * counts.clamp(min=1))
    return w
