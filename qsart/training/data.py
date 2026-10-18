"""Manifest adapter using exactly the same crop and normalization at inference."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from qsart.preprocessing import preprocess_image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_to_tensor(image: Image.Image, image_size: int = 224, mean=None, std=None) -> torch.Tensor:
    image = preprocess_image(image, image_size=image_size, crop=True)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    return (tensor - torch.tensor(mean or MEAN)[:, None, None]) / torch.tensor(std or STD)[:, None, None]


def split_fingerprint(frame: pd.DataFrame) -> str:
    # Hash row identities without placing identifiers or paths in exported metadata.
    values = sorted(str(Path(path).resolve()).casefold() for path in frame.image_path)
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path).resolve()
    frame = pd.read_csv(path, keep_default_na=True)
    if not {"image_path", "split"}.issubset(frame.columns):
        raise ValueError("Manifest needs image_path and split columns")
    if "annotation_status" in frame and not frame.annotation_status.eq("verified").all():
        raise ValueError("Manifest includes unverified or quarantined annotation_status records")
    for status_column in ("eligible_for_training", "readable", "exists"):
        if status_column in frame:
            accepted = frame[status_column].map(lambda value: str(value).strip().casefold() in {"true", "1"})
            if not accepted.all():
                raise ValueError(f"Manifest includes records with {status_column}=False or missing")
    frame["image_path"] = frame.image_path.map(
        lambda item: str((path.parent / str(item)).resolve()) if not Path(str(item)).is_absolute()
        else str(Path(str(item)).resolve()))
    if frame.image_path.str.casefold().duplicated().any():
        raise ValueError("Manifest contains duplicate image paths")
    for group_column in ("subject_id", "eye_id", "sha256"):
        if group_column in frame:
            groups = frame.dropna(subset=[group_column]).groupby(group_column)["split"].nunique()
            if (groups > 1).any():
                raise ValueError(f"Cross-split overlap in {group_column}")
    return frame


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, heads: dict[str, int], image_size: int = 224,
                 augment: bool = False, mean=None, std=None):
        self.heads = heads
        self.image_size = int(image_size)
        self.augment = augment
        self.mean, self.std = mean or MEAN, std or STD
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        for head in heads:
            if head not in frame:
                raise ValueError(f"Manifest lacks supervised label column: {head}")
            present = pd.to_numeric(frame[head], errors="coerce").dropna()
            if not ((present % 1 == 0) & (present >= 0) & (present < heads[head])).all():
                raise ValueError(f"Invalid class labels in {head}")
        self.frame = frame.loc[frame[list(heads)].notna().any(axis=1)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError("No labeled records in the requested split")
        for path in self.frame.image_path:
            if not Path(path).is_file():
                raise ValueError(f"Manifest image missing: {Path(path).name}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as handle:
            image = handle.convert("RGB")
        if self.augment and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        labels = {name: int(row[name]) if pd.notna(row[name]) else -100 for name in self.heads}
        return image_to_tensor(image, self.image_size, self.mean, self.std), labels
