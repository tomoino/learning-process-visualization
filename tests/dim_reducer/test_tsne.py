import pytest
from lpvis.dim_reducer.tsne import TSNE

@pytest.fixture
def tSNE() -> TSNE:
    return TSNE(
        output_dim=2,
        perplexity=1.0)

def test_fit_transform(tSNE, high_dim_df):
    low_dim_df = tSNE.fit_transform(high_dim_df)
    assert set(high_dim_df.columns) - set(low_dim_df.columns) == {"dim2"}