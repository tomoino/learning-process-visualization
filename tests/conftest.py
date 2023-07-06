import pytest
from polars import DataFrame
import random

@pytest.fixture
def high_dim_df() -> DataFrame:
    df = DataFrame(
        {
            "id": [i for i in range(9)],
            "t": [0,0,0,1,1,1,2,2,2],
            "dim0": [random.random() for _ in range(9)],
            "dim1": [random.random() for _ in range(9)],
            "dim2": [random.random() for _ in range(9)],
        }
    )
    return df
