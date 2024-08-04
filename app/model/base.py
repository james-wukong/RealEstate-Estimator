from pydantic import BaseModel, Field
from enum import Enum, unique


@unique
class OrderBy(str, Enum):
    ATV: str = "assdttlvalue"
    AVV: str = "avmvalue"
    BED: str = "beds"
    BT: str = "bathstotal"
    CD: str = "calendardate"
    LS1: str = "lotsize1"
    LS2: str = "lotsize2"
    PD: str = "publisheddate"
    PT: str = "propertytype"
    SA: str = "salesamt"
    SSD: str = "salesearchdate"
    STD: str = "saletransactiondate"
    US: str = "universalsize"


class SearchBase(BaseModel):
    orderby: OrderBy | None = Field(
        default=None,
        title="Sorting Options",
        max_length=25,
        examples=["universalsize"],
    )
    page: int = Field(
        default=1,
        gt=0,
        description="""The current view index based
        on the pagesize and the total
        number of records available""",
        examples=[1],
    )
    pagesize: int = Field(
        default=100,
        gt=0,
        description="""The number of records to be
        returned with the request""",
        examples=[1000],
    )


class ConfigModel(BaseModel):

    class Config:
        # allow_population_by_field_name = True
        populate_by_name = True
        extra = "ignore"
