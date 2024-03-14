import pytest
from lpvis.dim_reducer.dynamic_isne import DynamicISNE

@pytest.fixture
def dynamicISNE() -> DynamicISNE:
    return DynamicISNE(
        output_dim=2,
        time_oriented_penalty = 0.1, 
        last_structure_penalty = 0.1,
        learning_rate=0.1)

def test_fit_transform(dynamicISNE, high_dim_df):
    low_dim_df, references = dynamicISNE.fit_transform(high_dim_df)
    assert set(high_dim_df.columns) - set(low_dim_df.columns) == {"dim2"}