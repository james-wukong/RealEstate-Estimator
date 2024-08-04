# from functools import lru_cache
from typing import Any

from fastapi import APIRouter

from app.api.deps import (
    AddressLineDep,
    OrderbyDep,
    PageDep,
    PagesizeDep,
    PostcodeDep,
    ProptypeDep,
)
from app.api.utilities import get_avg_prices
from app.model.property import (
    PropertyResp,
    AddressReq,
    PropertyListResp,
)
from app.service.property import attom_property_api

# from app.api.deps import PostcodeDep

router = APIRouter()


@router.get(
    "/detail",
    response_model=PropertyListResp,
    response_model_include={"property"},
    summary="get property detail",
    description="""get property detail using expandedprofile
    by providing addresses in 2 lines,
    will add more fields based on requirement in future""",
)
async def get_property_detail(
    *,
    line1: AddressLineDep,
    line2: AddressLineDep,
) -> Any:
    """
    Retrieve items.
    """
    add_req = AddressReq(
        address1=line1,
        address2=line2,
    )
    resp = await attom_property_api(
        address=add_req,
        identifier="/property/expandedprofile",
    )

    return resp


# @lru_cache
@router.get(
    "/postal/{postal}",
    response_model=PropertyResp,
    summary="average house price by postal code",
    description="average unit price by postal code",
)
async def avg_by_postal(
    # address: AddressReq,
    *,
    postal: PostcodeDep,
    orderby: OrderbyDep = None,
    page: PageDep = 1,
    pagesize: PagesizeDep = 100,
    proptype: ProptypeDep = None,
) -> Any:
    """
    Retrieve items.
    """
    add_req = AddressReq(
        postalcode=postal,
        propertytype=proptype,
        page=page,
        orderby=orderby,
        pagesize=pagesize,
    )
    resp = await attom_property_api(address=add_req)
    size_t = resp.status.total

    result = PropertyResp(
        total=0,
        postal=postal,
        avg_assd_value=0,
        ttl_assd_value=0,
        avg_mkt_value=0,
        ttl_mkt_value=0,
    )
    result = await get_avg_prices(resp, result)

    assert result.total == size_t
    result.avg_assd_value = result.ttl_assd_value / result.total
    result.avg_mkt_value = result.ttl_mkt_value / result.total

    return result
