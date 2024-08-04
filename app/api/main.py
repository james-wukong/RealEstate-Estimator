from fastapi import APIRouter

from app.api.routes import property, estimator

api_router = APIRouter()
# api_router.include_router(login.router, tags=["login"])
api_router.include_router(
    property.router,
    prefix="/property",
    tags=["property"],
)
api_router.include_router(
    estimator.router,
    prefix="/estimator",
    tags=["estimator"],
)
