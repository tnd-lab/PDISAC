import base64
import gzip
import os
import sqlite3

import numpy as np
import pandas as pd
from configs.isac import N_block, N_chip


def decompress_matrix(b64_str, n_rows=N_chip, n_cols=N_block):
    """Invert MATLAB compressMatrix: base64 -> gunzip -> float32 -> complex64 column-major."""
    floats = np.frombuffer(gzip.decompress(base64.b64decode(b64_str)), dtype=np.float32)
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

    return df


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
