import warnings

from sklearn.manifold import TSNE as sklearnTSNE
import polars as pl
import numpy as np

from lpvis.dim_reducer.basic_dim_reducer import BasicDimReducer

def tsne(coords_df, t_column="t", output_dim=2, perplexity=30):
    """
    coords_df.columns = ["step", "dim0", "dim1", ...]
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)

    reduced_X = []
    steps = []
    max_step = coords_df[t_column].max()
    target_t_list = coords_df["t"].unique().to_list()
    
    for step in target_t_list:
        step_df = coords_df.filter(pl.col(t_column) == step).drop(t_column)
        _reduced_X = sklearnTSNE(n_components=output_dim, perplexity=perplexity).fit_transform(step_df.to_numpy())
        steps.extend([ float(step) for i in range(len(step_df))])
        reduced_X.extend(_reduced_X)
        
    steps_df = pl.DataFrame(
        data={t_column: steps}
    )

    reduced_X_df = pl.DataFrame(
        data=np.array(reduced_X, dtype='float64'),
        schema=[f"dim{i}" for i in range(output_dim)]
        )
    
    visible_coords_df = pl.concat([steps_df, reduced_X_df], how="horizontal")
    
    warnings.simplefilter(action='default', category=FutureWarning)

    return visible_coords_df


class TSNE(BasicDimReducer):
    def __init__(self, output_dim: int = 2, perplexity: float = 30.0, movement_penalty: float = 0.1):
        self.output_dim = output_dim
        self.perplexity = perplexity
        self.movement_penalty = movement_penalty
    
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fit df into an embedded space and return that transformed output.

        Args:
            df (pl.DataFrame): DataFrame to be transformed.
                "t" column is required.
                "dim0", "dim1", ... columns are required.
        """
        coords_df, meta_df = self.split_coords_and_meta(df)
        reduced_df = tsne(coords_df, perplexity=self.perplexity, output_dim=self.output_dim)

        return pl.concat([reduced_df, meta_df], how="horizontal")