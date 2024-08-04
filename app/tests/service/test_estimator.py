import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Generator
from xgboost import XGBRegressor

# from app.model.property import AddressReq
from app.service.estimator import load_model  # get_property_detail,

from app.model.estimator import AddressReq


@pytest.fixture
def address_req() -> AddressReq:
    return AddressReq(
        address1="4529 Winona Court",
        address2="Denver, CO",
    )


@pytest.fixture
def mock_joblib_load() -> Generator[MagicMock, None, None]:
    with patch("joblib.load") as mock_load:
        yield mock_load


def test_load_model(
    mock_joblib_load: MagicMock,
) -> None:
    # Prepare the mock object to return an instance of XGBRegressor
    mock_model = MagicMock(spec=XGBRegressor)
    mock_joblib_load.return_value = mock_model

    # Call the function to be tested
    filename = "xgb_model.sav"
    loaded_model = load_model(filename)

    # Assert the joblib.load function was called with the correct path
    expected_path = os.path.join("saved-models", filename)
    mock_joblib_load.assert_called_once_with(expected_path)

    # Assert the returned model is the mock model
    assert loaded_model == mock_model


# @pytest.mark.asyncio
# async def test_get_property_detail(address_req: AddressReq) -> None:
#     resp = await get_property_detail(address=address_req)

#     assert resp == ""
