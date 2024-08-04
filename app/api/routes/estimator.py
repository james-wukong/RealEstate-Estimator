from typing import Any
from fastapi import APIRouter

from app.api.deps import (
    AddressLineDep,
    # ProptypeDep,
)

from app.model.estimator import AddressReq, EstimatePriceModel
from app.service.estimator import get_property_est

router = APIRouter()


@router.get(
    "/price",
    response_model=EstimatePriceModel,
    summary="get property detail",
    description="""get property estimation by providing addresses in 2 lines,
    will add more fields based on requirement in future""",
)
async def get_estimate_price(
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
    est = await get_property_est(address=add_req)

    return est
