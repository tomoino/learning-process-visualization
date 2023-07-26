import polars as pl
import numpy as np

class GaussianShrinkCluster:
    def __init__(self, n_samples=100):
        self.df = self.generate(n_samples=100)

    def generate(self, n_samples=100) -> pl.DataFrame:
        # DataFrameに格納するための空のリストを作成
        coords = []

        # 3次元ガウス分布から300個の点を３クラスタサンプリング
        cov = np.eye(3)*0.1
        mean1 = np.array([1,0,0])
        points1 = np.random.multivariate_normal(mean1, cov, size=n_samples)

        mean2 = np.array([0,1,0])
        points2 = np.random.multivariate_normal(mean2, cov, size=n_samples)

        mean3 = np.array([0,0,1])
        points3 = np.random.multivariate_normal(mean3, cov, size=n_samples)

        # クラスタ1, 2, 3のデータを結合する
        points = np.concatenate([points1, points2, points3], axis=0)
        labels = [0]*len(points1) + [1]*len(points2) + [2]*len(points3)
        steps = [0 for i in range(points.shape[0])]

        coords.append(points)

        # 動かしつつDataFrameに追加
        for t in range(1, 11):
            alpha = 0.1 * t
            new_points1 = (1-alpha) * points1 + alpha * mean1
            new_points2 = (1-alpha) * points2 + alpha * mean2
            new_points3 = (1-alpha) * points3 + alpha * mean3
            new_points = np.concatenate([new_points1, new_points2, new_points3], axis=0)

            # points = np.concatenate([points, new_points], axis=0)
            labels.extend([0]*len(new_points1) + [1]*len(new_points2) + [2]*len(new_points3))
            steps.extend([t for i in range(new_points.shape[0])])
            coords.append(new_points)

        coords_ndarray = np.concatenate(coords, axis=0)
        ids = [i for i in range(coords_ndarray.shape[0])]
            
        df = pl.DataFrame(
            {
                "id": ids,
                "t": steps,
                "dim0": coords_ndarray[:, 0],
                "dim1": coords_ndarray[:, 1],
                "dim2": coords_ndarray[:, 2],
                "label": labels
            }
        )
            
        return df
