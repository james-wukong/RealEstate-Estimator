import pytest
from fastapi import HTTPException

# from app.model.property import AddressReq
from app.service import property as service_p
from app.model import property as model_p


@pytest.fixture
def postalcode_ok() -> str:
    return "82009"


@pytest.fixture
def postalcode_fail() -> str:
    return "82008"


@pytest.fixture
def address_req_ok(postalcode_ok: str) -> model_p.AddressReq:
    return model_p.AddressReq(
        postalcode=postalcode_ok,
        propertytype=None,
        page=1,
    )


@pytest.fixture
def address_req_fail(postalcode_fail: str) -> model_p.AddressReq:
    return model_p.AddressReq(
        postalcode=postalcode_fail,
        propertytype=None,
        page=1,
    )


@pytest.fixture
def profile_req_ok() -> model_p.AddressReq:
    return model_p.AddressReq(
        address1="4009 CARLA DR",
        address2="CHEYENNE, WY 82009",
    )


@pytest.fixture
def profile_req_fail() -> model_p.AddressReq:
    return model_p.AddressReq(
        address1="4009 CARLA DR",
        address2=None,
    )


@pytest.fixture
def api_address() -> str:
    return "/property/address"


@pytest.fixture
def api_profile() -> str:
    return "/property/basicprofile"


@pytest.mark.asyncio
async def test_attom_property_api_ok(
    address_req_ok: model_p.AddressReq,
) -> None:
    address_req_ok.address1 = None
    address_req_ok.address2 = None
    resp = await service_p.attom_property_api(
        address=address_req_ok,
    )
    assert resp.status.code == 0


@pytest.mark.asyncio
async def test_attom_property_api_fail(
    address_req_fail: model_p.AddressReq,
):
    address_req_fail.address1 = None
    address_req_fail.address2 = None
    with pytest.raises(HTTPException) as _:
        _ = await service_p.attom_property_api(
            address=address_req_fail,
        )
    # assert resp.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_attom_property_profile_ok(
    profile_req_ok: model_p.AddressReq,
    api_profile: str,
) -> None:
    resp = await service_p.attom_property_api(
        address=profile_req_ok,
        identifier=api_profile,
    )
    assert resp.status.code == 0, "wrong code in json"


# @pytest.mark.asyncio
# async def test_attom_property_profile_fail(profile_req_fail, api_profile):
#     resp = await service_p.attom_property_api(
#         address=profile_req_fail,
#         identifier=api_profile,
#     )
#     assert resp.status.code == 37, "wrong code in json"


@pytest.mark.asyncio
async def test_attom_property_profile_fail(
    profile_req_fail: model_p.AddressReq,
    api_profile: str,
) -> None:
    with pytest.raises(HTTPException) as _:
        _ = await service_p.attom_property_api(
            address=profile_req_fail,
            identifier=api_profile,
        )
