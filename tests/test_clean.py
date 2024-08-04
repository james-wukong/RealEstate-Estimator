import os
import pandas as pd
import pytest
from src.clean import DataClean


@pytest.fixture(name="fname")
def _filename() -> str:
    return os.path.join("data", "csv", "Query Results.csv")


@pytest.fixture(name="df")
def _dataframe(fname) -> pd.DataFrame:
    df = pd.read_csv(fname)
    return df


@pytest.fixture(name="obj_dc")
def _clean_object(fname) -> DataClean:
    obj = DataClean(file=fname)
    obj.replace_empty_brackets()
    return obj


def test_filename(fname) -> None:
    assert len(fname) > 0


def test_loaded(df: pd.DataFrame) -> None:
    assert df.empty is False
    assert "list_price" in df.columns.to_list()


def test_append_list(obj_dc: DataClean) -> None:
    obj_dc.removed_cols = []
    expected_str = ["first"]
    obj_dc.removed_cols = expected_str
    assert obj_dc.removed_cols == expected_str
    expected_list = ["a", "b", "c"]
    obj_dc.removed_cols = expected_list
    assert sorted(obj_dc.removed_cols) == sorted(expected_list + expected_str)


def test_remove_missing(obj_dc: DataClean) -> None:
    miss_perc = 90
    expected_removed = "additional_parcels_description"
    expected_remain = "association_name"
    expected_not_100 = "building_area_source"
    assert obj_dc.dataframe[expected_not_100].isna().sum() > 0
    obj_dc.remove_missing(miss_perc=miss_perc)
    # assert obj_dc.dataframe[expected_not_100].isna().sum() == 0
    assert expected_removed not in obj_dc.dataframe.columns
    assert expected_remain in obj_dc.dataframe.columns


def test_update_association_yn(obj_dc: DataClean) -> None:
    row1 = "4c5b08c1-f67f-462a-8511-7ca530222814"
    row2 = "5d88bac7-ee6d-4385-817b-416aa5e9222d"
    test_fields = [
        "association_fee_frequency",
        "association_fee",
        "association_fee2_frequency",
        "association_fee2",
        "association_name",
        "association_name2",
        "association_phone",
        "association_phone2",
    ]
    obj_dc.update_association_yn(
        fields=test_fields,
        target_col="association_yn",
    )
    df_row = obj_dc.dataframe.loc[obj_dc.dataframe["id"] == row1]
    df_row2 = obj_dc.dataframe.loc[obj_dc.dataframe["id"] == row2]

    assert not df_row["association_yn"].values[0], "value is true"
    assert df_row2["association_yn"].values[0], "value is false"


def test_merg_association_cols(obj_dc: DataClean) -> None:
    freq_col: list[str] = [
        "association_fee_frequency",
        "association_fee2_frequency",
    ]
    fee_col: list[str] = ["association_fee", "association_fee2"]
    new_col: str = "association_fee_t"
    expected_fee: float = 1832.72
    obj_dc.merg_association_cols(
        freq_cols=freq_col,
        fee_cols=fee_col,
        new_col=new_col,
    )
    row = obj_dc.dataframe[
        obj_dc.dataframe["id"] == "19a87375-960b-4f35-a1d6-0b5ec83a2e26"
    ]
    assert (
        row[new_col].values[0] == expected_fee
    ), f"result not equal {row[new_col].values[0]}"
    assert (
        "association_fee_frequency" not in obj_dc.dataframe.columns
    ), "association_fee_frequency in columns"
    assert new_col in obj_dc.dataframe.columns, new_col + "not in columns"


# def test_with_fail():
#     pytest.fail("failed test")
