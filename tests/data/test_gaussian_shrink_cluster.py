from lpvis.data.gaussian_shrink_cluster import GaussianShrinkCluster

def test_gaussian_shrink_cluster():
    gsc = GaussianShrinkCluster()
    assert set(gsc.df.columns) == {"id", "t", "dim0", "dim1", "dim2", "label"}
    assert gsc.df.shape == (3300, 6)