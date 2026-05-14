import pandas as pd
import pytest

from readmission.data.validate import validate_columns


def test_validate_columns_raises_for_missing_target():
    with pytest.raises(ValueError):
        validate_columns(pd.DataFrame({"age": ["[50-60)"]}), {"readmitted"})

