from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from typing import Optional
import pandas as pd
from utils.utils import convert_from_complex


def normalize_rd(Z_rd, eps=1e-8):
    # Mean and std over H and W for each batch and channel
    mean = Z_rd.mean(axis=(2, 3), keepdims=True)
    std = Z_rd.std(axis=(2, 3), keepdims=True)

    Z_norm = (Z_rd - mean) / (std + eps)

    return Z_norm

class RDMapDataset(Dataset):
    """
    Paired (noisy, clean) RD map dataset for denoising.

    Args:
        P_rd : (N, 2, H, W) float32
        P_rd_prbs : (N, 2, H, W) float32
        gids     : (N,)  RAN_Index values — used for GroupKFold
    """

    def __init__(
        self,
        P_rd: np.ndarray,
        P_rd_prbs: np.ndarray,
        gids: np.ndarray,
    ):
        assert len(P_rd) == len(P_rd_prbs) == len(gids)
        self.P_rd = P_rd
        self.P_rd_prbs = P_rd_prbs
        self.gids = gids

    @classmethod
    def build(cls, df: pd.DataFrame, rows: Optional[int] = None) -> "RDMapDataset":
        """
        Build from a DataFrame produced by load_radar_results().

        Required columns: RD_map, RD_map_no_noise, RAN_Index.

        Args:
            df   : full DataFrame
            rows : cap on number of samples (None = all)
        """
        df = df[df["RD_map"].notna() & df["RD_map_no_noise"].notna()].reset_index(
            drop=True
        )

        if rows is not None:
            df = df.head(rows)

        P_rd = np.stack([convert_from_complex(m) for m in df["RD_map"]])
        P_rd_prbs = np.stack(
            [convert_from_complex(m) for m in df["RD_map_no_noise"]]
        )

        P_rd = normalize_rd(P_rd)
        P_rd_prbs = normalize_rd(P_rd_prbs)

        gids = df["RAN_Index"].to_numpy()

        return cls(P_rd, P_rd_prbs, gids)

    def __len__(self) -> int:
        return len(self.gids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return (
            torch.from_numpy(self.P_rd[idx]),
            torch.from_numpy(self.P_rd_prbs[idx]),
        )

    @property
    def map_shape(self) -> tuple[int, int, int]:
        """Shape of one sample: (2, H, W)."""
        return tuple(self.P_rd.shape[1:])


class RDMapDataloader:
    """
    Wraps RDMapDataset with GroupKFold splitting on RAN_Index.

    GroupKFold (not Stratified) because there are no class labels —
    the only constraint is that correlated trials (same RAN_Index)
    never appear in both train and val.

    Usage:
        loader = RDMapDataloader(df, n_splits=5, batch_size=32)
        for train_loader, val_loader in loader.get_fold_dataloaders():
            for P_rd, P_rd_prbs in train_loader:
                # P_rd, P_rd_prbs : (B, 2, H, W) float32
                ...

    Args:
        df           : DataFrame from load_radar_results()
        rows         : max samples to use (None = all)
        n_splits     : number of CV folds
        batch_size   : DataLoader batch size
        num_workers  : DataLoader worker processes
        pin_memory   : passed to DataLoader
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rows: Optional[int] = None,
        n_splits: int = 5,
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset = RDMapDataset.build(df, rows=rows)
        print(
            f"Dataset built: {len(self.dataset)} samples | "
            f"map shape: {self.dataset.map_shape} | "
            f"unique RAN_Index groups: {len(np.unique(self.dataset.gids))}"
        )

        kfold = GroupKFold(n_splits=n_splits)
        X = np.arange(len(self.dataset))
        groups = self.dataset.gids

        self.fold_indices = [
            (train_idx, val_idx) for train_idx, val_idx in kfold.split(X, groups=groups)
        ]
        print(f"Computed {n_splits} folds using GroupKFold on RAN_Index.")

    def get_fold_dataloaders(self) -> list[tuple[DataLoader, DataLoader]]:
        """
        Returns [(train_loader, val_loader), ...] — one pair per fold.

        Each loader yields (P_rd, P_rd_prbs) batches of shape (B, 2, H, W).
        """
        fold_loaders = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.fold_indices):
            train_loader = DataLoader(
                Subset(self.dataset, train_idx),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            val_loader = DataLoader(
                Subset(self.dataset, val_idx),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            fold_loaders.append((train_loader, val_loader))
            print(
                f"  Fold {fold_idx + 1}: "
                f"train={len(train_idx):4d}  val={len(val_idx):4d}"
            )

        return fold_loaders

    @property
    def map_shape(self) -> tuple[int, int, int]:
        return self.dataset.map_shape


if __name__ == "__main__":
    from data.read_data import load_radar_results
    from configs.data import db_path

    df = load_radar_results(db_path, n_records=10)

    loader = RDMapDataloader(df, n_splits=3, batch_size=1)

    for fold_idx, (train_loader, val_loader) in enumerate(
        loader.get_fold_dataloaders()
    ):
        P_rd, P_rd_prbs = next(iter(train_loader))
        print(
            f"  Fold {fold_idx + 1} | "
            f"noisy: {tuple(P_rd.shape)} | "
            f"P_rd_prbs: {tuple(P_rd_prbs.shape)} | "
            f"dtype: {P_rd.dtype}"
        )

    print("Passed.")
