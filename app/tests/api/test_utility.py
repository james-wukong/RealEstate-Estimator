import pytest
from app.api.utilities import get_avg_price
from app.model.property import AddressReq, PropertyResp


@pytest.fixture
def address_req() -> AddressReq:
    return AddressReq(
        address1="4009 CARLA DR",
        address2="CHEYENNE, WY 82009",
    )


@pytest.fixture
def property_init() -> PropertyResp:
    return PropertyResp(
        total=0,
        postal="82009",
        avg_assd_value=0,
        avg_mkt_value=0,
        ttl_assd_value=0,
        ttl_mkt_value=0,
    )


@pytest.fixture
def api_profile() -> str:
    return "/property/basicprofile"


@pytest.mark.asyncio
async def test_get_avg_price(
    address_req: AddressReq,
    property_init: PropertyResp,
):
    result = await get_avg_price(
        detail_req=address_req,
        result=property_init,
    )
    assert result.ttl_assd_value == 28366
