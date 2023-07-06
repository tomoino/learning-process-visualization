from typing import Tuple
import polars as pl

class BasicDimReducer:
    def __init__(self):
        pass

    def split_coords_and_meta(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        次元削減に使う情報とメタ情報を分離する
        """
        coords_columns = [col for col in df.columns if col.startswith("dim") or col == "t"]
        meta_columns = [col for col in df.columns if col not in coords_columns]

        coords_df = df[coords_columns]
        meta_df = df[meta_columns]

        return coords_df, meta_df