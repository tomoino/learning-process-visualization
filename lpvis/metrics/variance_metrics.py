from polars import DataFrame
import numpy as np
from typing import Dict, List

class VarianceMetrics:
    def __init__(self):
        pass

    def trace_covmat(self, df: DataFrame, val_columns=["dim0", "dim1"], t_column="t", label_column="label", metrics_col="trace_cov") -> DataFrame:
        """
        時刻ごと・クラスタごとに共分散行列のトレースを計算し、
        時刻ごとに平均をとる
        """

        # 時刻ごとに共分散行列を計算
        covmat_df_group = df.groupby(t_column, label_column)
        trace_cov_dict: Dict[int, List[float]] = {}
        for _, data in covmat_df_group:  
            t = data[t_column][0]
            if t not in trace_cov_dict:
                trace_cov_dict[t] = []
            cov = np.trace(np.cov(np.transpose(data.select(val_columns))))
            
            trace_cov_dict[t].append(cov)
        df = DataFrame(
            {
                "t": [t for t in trace_cov_dict.keys()],
                metrics_col: [np.mean(trace_cov_dict[t]) for t in trace_cov_dict.keys()]
            }
        )

        return df