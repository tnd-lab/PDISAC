"""
Loader for the AFM training scripts (scripts/train/train_pdnet_afm.py, train_dncnn_afm.py).

Yields (Z_rd, Z_rd_prbs, Z_rd_alt, M_tars), where Z_rd_alt is a second independent
realisation of the same scene (other data symbols and noise) written by the MATLAB
dataset generator into the column matrix_data.RD_map_alt_compressed. The scripts
without AFM use data/dataloader.py, which needs only the first two maps.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold

from data.read_data import decompress_matrix, attach_target_truth, target_support_mask
from data.dataloader import normalize_rd
from utils.utils import convert_from_complex


def load_radar_results_afm(db_file: str, n_records: Optional[int] = None) -> pd.DataFrame:
    """As load_radar_results(), plus the RD_map_alt column (second realisation of the scene)."""
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f"Database not found: '{db_file}'")

    cols = ("m.RD_map_compressed, m.RD_map_no_noise_compressed, m.RD_map_alt_compressed")
    if n_records is not None:
        sql = f"""
            WITH ranked AS (
                SELECT rowid,
                       ROW_NUMBER() OVER (PARTITION BY RAN_Index ORDER BY Trial_ID) AS rn
                FROM   scalar_results
            )
            SELECT s.*, {cols}
            FROM   scalar_results s
            JOIN   matrix_data    m ON m.Trial_ID = s.Trial_ID
            JOIN   ranked         r ON r.rowid    = s.rowid
            WHERE  r.rn <= {n_records}
        """
    else:
        sql = f"""
            SELECT s.*, {cols}
            FROM   scalar_results s
            JOIN   matrix_data    m ON m.Trial_ID = s.Trial_ID
        """

    with sqlite3.connect(db_file) as conn:
        try:
            df = pd.read_sql_query(sql, conn)
        except pd.errors.DatabaseError as exc:
            raise RuntimeError(
                "RD_map_alt_compressed not found in matrix_data. The AFM scripts need a "
                "database that also stores the second realisation Z_rd_alt of each scene; "
                "a database with only two RD-map columns serves the no-AFM scripts."
            ) from exc

    df["RD_map"] = df["RD_map_compressed"].apply(decompress_matrix)
    df["RD_map_no_noise"] = df["RD_map_no_noise_compressed"].apply(decompress_matrix)
    df["RD_map_alt"] = df["RD_map_alt_compressed"].apply(decompress_matrix)
    df.drop(
        columns=[
            "RD_map_compressed",
            "RD_map_no_noise_compressed",
            "RD_map_alt_compressed",
        ],
        inplace=True,
    )
    return attach_target_truth(df, db_file)


class RDMapDatasetAFM(Dataset):
    """(noisy, clean, alternative-noisy) triplets; same normalisation as RDMapDataset."""

    def __init__(self, P_rd, P_rd_prbs, P_rd_alt, gids, truth):
        assert len(P_rd) == len(P_rd_prbs) == len(P_rd_alt) == len(gids) == len(truth)
        self.truth = truth                              # per-sample [[range_m, vel_mps], ...]
        self.P_rd = P_rd
        self.P_rd_prbs = P_rd_prbs
        self.P_rd_alt = P_rd_alt
        self.gids = gids

    @classmethod
    def build(cls, df: pd.DataFrame, rows: Optional[int] = None) -> "RDMapDatasetAFM":
        df = df[
            df["RD_map"].notna()
            & df["RD_map_no_noise"].notna()
            & df["RD_map_alt"].notna()
        ].reset_index(drop=True)
        if rows is not None:
            df = df.head(rows)

        P_rd = normalize_rd(np.stack([convert_from_complex(m) for m in df["RD_map"]]))
        P_rd_prbs = normalize_rd(
            np.stack([convert_from_complex(m) for m in df["RD_map_no_noise"]])
        )
        # The alternative map is normalised with its OWN statistics, exactly like
        # Z_rd, so the two mixup endpoints live on the same scale.
        P_rd_alt = normalize_rd(
            np.stack([convert_from_complex(m) for m in df["RD_map_alt"]])
        )
        return cls(P_rd, P_rd_prbs, P_rd_alt, df["RAN_Index"].to_numpy(), list(df["truth"]))

    def __len__(self) -> int:
        return len(self.gids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            torch.from_numpy(self.P_rd[idx]),
            torch.from_numpy(self.P_rd_prbs[idx]),
            torch.from_numpy(self.P_rd_alt[idx]),
            torch.from_numpy(target_support_mask(self.P_rd_prbs[idx], self.truth[idx])),   # M_tars
        )

    @property
    def map_shape(self) -> tuple[int, int, int]:
        return tuple(self.P_rd.shape[1:])


class RDMapDataloaderAFM:
    """GroupKFold on RAN_Index, same interface as RDMapDataloader."""

    def __init__(self, df, n_splits=5, batch_size=1, num_workers=1,
                 pin_memory=False, rows=None):
        self.dataset = RDMapDatasetAFM.build(df, rows=rows)
        print(
            f"Dataset (AFM) built: {len(self.dataset)} samples | "
            f"map shape {self.dataset.map_shape}"
        )
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_fold_dataloaders(self) -> list[tuple[DataLoader, DataLoader]]:
        gkf = GroupKFold(n_splits=self.n_splits)
        idx = np.arange(len(self.dataset))
        folds = []
        for tr, va in gkf.split(idx, groups=self.dataset.gids):
            folds.append(
                (
                    DataLoader(
                        Subset(self.dataset, tr),
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                    ),
                    DataLoader(
                        Subset(self.dataset, va),
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                    ),
                )
            )
        return folds
