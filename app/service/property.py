# import json
from fastapi import HTTPException, status

from httpx import AsyncClient

from app.core.config import settings
from app.model.property import AddressReq, PropertyListResp


async def attom_property_api(
    address: AddressReq,
    identifier: str = "/property/address",
) -> PropertyListResp:
    """ATTOM PROPERTY API

    Args:
        address (AddressReq): _description_
        identifier (str, optional): _description_.
        Defaults to "/property/address".

    Raises:
        Exception: _description_
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        Response: _description_
    """
    if settings.ATTOM_ENDPOINT and settings.ATTOM_TOKEN:
        api_url = "/".join(
            part.strip("/")
            for part in [
                settings.ATTOM_ENDPOINT,
                identifier,
            ]
        )
    else:
        raise Exception("ATTOM_ENDPOINT or ATTOM_ENDPOINT not set yet")
    headers = {
        "accept": "application/json",
        "apikey": settings.ATTOM_TOKEN,
    }
    async with AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            api_url,
            params=address.model_dump(exclude_none=True),
            headers=headers,
        )
    # print(resp.request.url)
    if resp.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"response status: {resp.status_code}",
        )
    if not resp.json():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Content not found"
        )
    # with open("test.json", "wb") as file:
    #     file.write(json.dumps(resp.json(), indent=4).encode("utf-8"))
    return PropertyListResp(**resp.json())
