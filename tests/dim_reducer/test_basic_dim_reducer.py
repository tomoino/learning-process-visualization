import pytest
from lpvis.dim_reducer.basic_dim_reducer import BasicDimReducer

@pytest.fixture
def basicDimReducer() -> BasicDimReducer:
    return BasicDimReducer()

def test_split_coords_and_meta(basicDimReducer, high_dim_df):
    coords, meta = basicDimReducer.split_coords_and_meta(high_dim_df)
    assert coords.columns == ["t", "dim0", "dim1", "dim2"]
    assert meta.columns == ["id"]