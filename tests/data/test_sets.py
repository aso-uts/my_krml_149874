import pytest
import pandas as pd

from my_krml_149874.data.sets import pop_target


@pytest.fixture
def features_fixture():
    features_data = [
        [1, 25, "Junior"],
        [2, 33, "Confirmed"],
        [3, 42, "Manager"],
    ]
    return pd.DataFrame(features_data, columns=["employee_id", "age", "level"])

@pytest.fixture
def target_fixture():
    target_data = [5, 10, 20]
    return pd.Series(target_data, name="salary", copy=False)

def test_pop_target_with_data_fixture(features_fixture, target_fixture):
    input_df = features_fixture.copy()
    input_df["salary"] = target_fixture
    
    features, target = pop_target(df=input_df, target_col='salary')

    pd.testing.assert_frame_equal(features, features_fixture)
    pd.testing.assert_series_equal(target, target_fixture)

def test_pop_target_no_col_found(features_fixture, target_fixture):
    input_df = features_fixture.copy()
    
    with pytest.raises(KeyError):
        features, target = pop_target(df=input_df, target_col='salary')

def test_pop_target_col_none(features_fixture, target_fixture):
    input_df = features_fixture.copy()
    
    with pytest.raises(KeyError):
        features, target = pop_target(df=input_df, target_col=None)

def test_pop_target_df_none(features_fixture, target_fixture):
    input_df = features_fixture.copy()
    
    with pytest.raises(AttributeError):
        features, target = pop_target(df=None, target_col="salary")