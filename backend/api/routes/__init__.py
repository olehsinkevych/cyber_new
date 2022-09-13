from fastapi import APIRouter
from backend.api.routes.measurements import router as data_temperature


router = APIRouter()
router.include_router(data_temperature, prefix="", tags=["measurements"])