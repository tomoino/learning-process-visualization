from lpvis.metrics.variance_metrics import VarianceMetrics

def test_variance_metrics(low_dim_df):
    varmet = VarianceMetrics()
    tr_covmat_df = varmet.trace_covmat(low_dim_df, metrics_col="test", val_columns=["dim0", "dim1"], t_column="t", label_column="label")

    assert tr_covmat_df.shape == (3, 2)

