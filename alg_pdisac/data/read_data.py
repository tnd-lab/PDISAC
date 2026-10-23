import base64
import gzip
import os
import sqlite3

import numpy as np
import pandas as pd
from configs.isac import N_block, N_chip, range_resolution, vel_axis


def decompress_matrix(blob, n_rows=N_chip, n_cols=N_block):
    """Invert MATLAB generate_radar_dataset: gunzip -> float32 -> complex64 column-major.

    Accepts the BLOB (bytes) written by the current generator and, for older
    databases, the base64 text of the previous format; both are lossless.
    """
    raw = bytes(blob) if isinstance(blob, (bytes, bytearray, memoryview)) else base64.b64decode(blob)
    floats = np.frombuffer(gzip.decompress(raw), dtype=np.float32)
    n_el = n_rows * n_cols
    return (
        floats[:n_el].reshape(n_rows, n_cols, order="F")
        + 1j * floats[n_el:].reshape(n_rows, n_cols, order="F")
    ).astype(np.complex64)


def inspect_db(db_file="dataset_train.db"):
    """Print tables and row counts — useful for verifying the DB path and contents."""
    if not os.path.isfile(db_file):
        print(f"File not found: {db_file}")
        return
    print(f"DB: {db_file}  ({os.path.getsize(db_file) / 1e6:.1f} MB)")
    with sqlite3.connect(db_file) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        print(f"Tables: {tables}")
        for t in tables:
            n = pd.read_sql_query(f"SELECT COUNT(*) as n FROM {t}", conn)["n"][0]
            print(f"  {t}: {n} rows")


def load_radar_results(db_file="dataset_train.db", n_records=None):
    if not os.path.isfile(db_file):
        raise FileNotFoundError(
            f"Database not found: '{db_file}'\n"
            "Run MATLAB dataset_train first to generate dataset_train.db, "
            "then pass its path via db_file."
        )

    # limit = f"LIMIT {n_records}" if n_records is not None else ""
    # n_records per unique RAN_Index, not total rows — adjust as needed for your use case

    if n_records is not None:
        sql = f"""
            WITH ranked AS (
                SELECT rowid,
                    ROW_NUMBER() OVER (
                        PARTITION BY RAN_Index
                        ORDER BY Trial_ID
                    ) AS rn
                FROM   scalar_results
            )
            SELECT s.*, m.RD_map_compressed, m.RD_map_no_noise_compressed
            FROM   scalar_results s
            JOIN   matrix_data    m ON m.Trial_ID = s.Trial_ID
            JOIN   ranked         r ON r.rowid    = s.rowid
            WHERE  r.rn <= {n_records}
        """
    else:
        sql = """
            SELECT s.*, m.RD_map_compressed, m.RD_map_no_noise_compressed
            FROM   scalar_results s
            JOIN   matrix_data    m ON m.Trial_ID = s.Trial_ID
        """

    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query(sql, conn)

    df["RD_map"] = df["RD_map_compressed"].apply(decompress_matrix)
    df["RD_map_no_noise"] = df["RD_map_no_noise_compressed"].apply(decompress_matrix)

    df.drop(columns=["RD_map_compressed", "RD_map_no_noise_compressed"], inplace=True)

    return attach_target_truth(df, db_file)


if __name__ == "__main__":
    from configs.data import db_path

    inspect_db(db_path)
    df_all = load_radar_results(db_path)

    # plot RD maps for the first record
    import matplotlib.pyplot as plt
    from utils.utils import convert_to_db, convert_from_complex
    from utils.plots import plot_rd

    Z_rd = df_all["RD_map"].iloc[0]
    Z_rd_prbs = df_all["RD_map_no_noise"].iloc[0]

    # print unique RAN_Index and it counts
    print("Unique RAN_Index values and counts:")
    print(df_all["RAN_Index"].value_counts())

    Z_rd = convert_from_complex(Z_rd)
    Z_rd_prbs = convert_from_complex(Z_rd_prbs)

    P_rd = np.abs(Z_rd) ** 2
    P_rd_prbs = np.abs(Z_rd_prbs) ** 2

    P_rd = convert_to_db(P_rd)
    P_rd_no_noise = convert_to_db(P_rd_prbs)

    plot_rd(P_rd_no_noise[0])
    plot_rd(P_rd_no_noise[1])


# --------------------------------------------------------------------------- #
#  Target-support mask M_tars (training only; not used at inference)
# --------------------------------------------------------------------------- #
# Half-width (bins) of the mainlobe of a target response in the RD map: no range or
# Doppler window, so the first nulls lie 1 bin from the true position and every
# mainlobe cell is within +-1 bin of the rounded true cell (independent of CA-CFAR).
MAINLOBE_HALF_WIDTH = 1


def attach_target_truth(df, db_file):
    """Column 'truth': array [[range_m, vel_mps], ...] of the targets of each scene (table 'targets')."""
    with sqlite3.connect(db_file) as conn:
        tg = pd.read_sql_query("SELECT RAN_Index, True_Range_m, True_Vel_mps FROM targets", conn)
    truth = {k: g[["True_Range_m", "True_Vel_mps"]].to_numpy() for k, g in tg.groupby("RAN_Index")}
    empty = np.zeros((0, 2))
    df["truth"] = [truth.get(k, empty) for k in df["RAN_Index"]]
    return df


def target_support_mask(Z_prbs, truth, w=MAINLOBE_HALF_WIDTH):
    """
    M_tars (2, N_chip, N_block) in [0,1] (manuscript IV-C): the clean RD power
    P = |Z_prbs|^2 is normalized per target over its mainlobe, so that every target
    peak maps to unity and all cells outside the target mainlobes to zero.
    Z_prbs: (2, N_chip, N_block) real/imag clean map;  truth: [[range_m, vel_mps], ...].
    Cells: k = round(r / range_resolution), d = argmin |vel_axis - v| (as in main5b).
    """
    P = Z_prbs[0] ** 2 + Z_prbs[1] ** 2
    M = np.zeros_like(P)
    d0 = int(np.argmin(np.abs(vel_axis)))                  # removed zero-Doppler column
    for r, v in np.asarray(truth).reshape(-1, 2):
        k = int(round(r / range_resolution))
        d = int(np.argmin(np.abs(vel_axis - v)))
        if d == d0 or k < 0 or k >= P.shape[0]:
            continue                                        # not visible in the RD map
        box = (slice(max(0, k - w), k + w + 1), slice(max(0, d - w), d + w + 1))
        peak = P[box].max()
        if peak > 0:
            M[box] = np.maximum(M[box], P[box] / peak)
    return np.stack([M, M]).astype(np.float32)
