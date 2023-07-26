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
            "label": [0,1,2,0,1,2,0,1,2]
        }
    )
    return df

@pytest.fixture
def low_dim_df() -> DataFrame:
    df = DataFrame(
        {
            "id": [i for i in range(18)],
            "t": [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2],
            "dim0": [random.random() for _ in range(18)],
            "dim1": [random.random() for _ in range(18)],
            "label": [0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1,2,2]
        }
    )
    return df
