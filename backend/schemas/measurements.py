from typing import Optional
from datetime import datetime

from pydantic import BaseModel


class TemperatureBase(BaseModel):
    time_point: datetime
    user: Optional[str] = None
    value: float
    tag: Optional[str] = None


class TemperatureCreate(TemperatureBase):
    pass


class TemperatureGet(TemperatureBase):
    id: int

    class Config:
        orm_mode = True