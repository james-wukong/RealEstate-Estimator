# import asyncio
# from functools import lru_cache

import os
import pandas as pd
from app.model.property import AddressReq, PropertyListResp, PropertyResp
from app.service.property import attom_property_api


# @lru_cache
async def get_avg_prices(
    resp: PropertyListResp,
    result: PropertyResp,
) -> PropertyResp:
    for key, props in resp.model_dump().items():
        if key == "property":
            for details in props:
                if (
                    details["address"]
                    and details["address"]["line1"]
                    and details["address"]["line2"]
                ):
                    detail_req = AddressReq(
                        address1=details["address"]["line1"],
                        address2=details["address"]["line2"],
                    )
                    result = await get_avg_price(
                        detail_req,
                        result,
                    )
    return result


# @lru_cache
async def get_avg_price(
    detail_req: AddressReq,
    result: PropertyResp,
) -> PropertyResp:
    detail_resp = await attom_property_api(
        address=detail_req,
        identifier="/property/basicprofile",
    )
    # lock = asyncio.Lock()
    for key, props in detail_resp.model_dump().items():
        if key == "property":
            # async with lock:
            for details in props:
                if (
                    details["assessment"]["assessed"]
                    and details["assessment"]["market"]
                    and details["assessment"]["assessed"]["assd_ttl_value"]
                    and details["assessment"]["market"]["mkt_ttl_value"]
                ):
                    result.total += 1
                    result.ttl_assd_value += details["assessment"]["assessed"][
                        "assd_ttl_value"
                    ]
                    result.ttl_mkt_value += details["assessment"]["market"][
                        "mkt_ttl_value"
                    ]
    return result


def init_dataframe(
    filename: str,
    basepath: str = "app/data/csv",
) -> pd.DataFrame:
    return pd.read_csv(os.path.join(basepath, filename))
