from fastapi.testclient import TestClient
import pytest

# import json

from app.core.config import settings


@pytest.fixture
def address1() -> str:
    return "4529 Winona Court"


@pytest.fixture
def address2() -> str:
    return "Denver, CO"


def test_get_detail_api(
    client: TestClient,
    address1: str,
    address2: str,
) -> None:
    params = {"line1": address1, "line2": address2}
    response = client.get(
        url=f"{settings.API_V1_STR}/property/detail",
        headers=None,
        params=params,
    )

    assert response.status_code == 200
