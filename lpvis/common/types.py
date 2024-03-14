import polars as pl
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset

class Coords:
    def __init__(self, coords: List[np.ndarray], ids: List[int] = None):
        self.df = self.make_df_from_ndarrays(coords, ids)
        self.coords = coords
        self.dim = coords[0].shape[1]
        """
        Coords
            id: int
            dim0: float
            dim1: float
            ...
            label: int
        """

    def make_df_from_ndarrays(self, ndarrays: List[np.ndarray], ids: List[int] = None) -> pl.DataFrame:
        
        df_dict = {}

        # labels の作成
        labels = []
        for i, ndarray in enumerate(ndarrays):
            labels.extend([i]*len(ndarray))
        df_dict["label"] = labels

        # すべての ndarray を縦に結合
        coords_ndarray = np.concatenate(ndarrays, axis=0)
        for i in range(coords_ndarray.shape[1]):
            df_dict[f"dim{i}"] = coords_ndarray[:, i]

        # id の作成
        if ids is None:
            ids = [i for i in range(coords_ndarray.shape[0])]
            
        df_dict["id"] = ids

        df = pl.DataFrame(
            df_dict
        )

        df = df.sort("id")
            
        return df
    
    def to_dataset(self) -> Dataset:
        # df の dim{i} を座標として持ち、label をラベルとして持つデータセットを作成
        # カスタムのTorchデータセットを作成
        class CustomDataset(Dataset):
            def __init__(self, coords: np.ndarray, labels: np.ndarray):
                self.coords = torch.tensor(coords, dtype=torch.float32)
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self):
                return len(self.coords)

            def __getitem__(self, idx):
                return self.coords[idx], self.labels[idx]
        
        dataset = CustomDataset(
            coords=self.df[[f"dim{i}" for i in range(self.dim)]].to_numpy(),
            labels=self.df["label"].to_numpy()
        )
        return dataset
    
class TimeSeries:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        """
        TimeSeries
            id: int
            dim0: float
            dim1: float
            ...
            label: int
            t: int
        """