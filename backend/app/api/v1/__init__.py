"""API v1 package."""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.trip import router as trip_router
from app.api.v1.trip_streaming import router as trip_streaming_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(trip_router)
api_router.include_router(trip_streaming_router)
