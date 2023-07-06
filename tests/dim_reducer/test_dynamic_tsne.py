import pytest
from lpvis.dim_reducer.dynamic_tsne import DynamicTSNE

@pytest.fixture
def dynamicTSNE() -> DynamicTSNE:
    return DynamicTSNE(
        output_dim=2,
        perplexity=30.0,
        movement_penalty=0.1)

def test_split_coords_and_meta(dynamicTSNE, high_dim_df):
    coords, meta = dynamicTSNE.split_coords_and_meta(high_dim_df)
    assert coords.columns == ["t", "dim0", "dim1", "dim2"]
    assert meta.columns == ["id"]

def test_fit_transform(dynamicTSNE, high_dim_df):
    low_dim_df = dynamicTSNE.fit_transform(high_dim_df)
    assert set(low_dim_df.columns) == {"t", "dim0", "dim1", "id"}