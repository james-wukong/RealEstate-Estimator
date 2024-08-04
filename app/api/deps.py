from typing import Annotated

from fastapi import Path, Query

from app.model.property import PropertyType
from app.model.base import OrderBy

# from app.core.config import settings

PostcodeDep = Annotated[
    str,
    Path(..., title="postal code", max_length=50, regex="^[a-zA-Z0-9]+$"),
]
PageDep = Annotated[int, Query(..., title="current page", gt=0)]
PagesizeDep = Annotated[
    int,
    Query(..., title="total pages", gt=0),
]
OrderbyDep = Annotated[OrderBy, Query(..., title="sorting options")]
ProptypeDep = Annotated[PropertyType, Query(..., title="property types")]

AddressLineDep = Annotated[
    str,
    Query(
        ...,
        title="property types",
        max_length=50,
        regex="^[a-zA-Z0-9\s,'.-/]+$",
        description="The address line, including street number, name, etc.",
    ),
]

# StmtIntervalDep = Annotated[
#     StatementInterval,
#     Query(..., title="interval of statement"),
# ]
