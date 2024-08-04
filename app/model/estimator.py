# from enum import Enum, unique
from pydantic import BaseModel, Field

# from app.model.base import SearchBase, ConfigModel


class AddressReq(BaseModel):
    address1: str = Field(
        default="",
        title="The first line of the property address",
        max_length=100,
    )
    address2: str = Field(
        default="",
        title="The second line of the property address",
        max_length=100,
    )


class EstimatePriceModel(BaseModel):
    est_price: float = 0
    mkt_price: float = 0
    assd_price: float = 0
    sale_price: float = 0
