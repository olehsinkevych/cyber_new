from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends
from fastapi import status
from sqlalchemy.orm.session import Session
from fastapi import HTTPException
from backend.schemas.measurements import TemperatureGet, TemperatureCreate
from backend.api.dependencies import get_db
from backend.db.repositories.crud import TemperatureCRUD

router = APIRouter(include_in_schema=True)


@router.get("/temperature/{id}", response_model=TemperatureGet,
            status_code=status.HTTP_200_OK)
def read_on_id(temp_id: int, db: Session = Depends(get_db)):
    crud = TemperatureCRUD(db_session=db)
    records = crud.read_on_id(id_=temp_id)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Item not found")
    return records


@router.post("/temperature/", response_model=TemperatureGet,
             status_code=status.HTTP_200_OK)
def create(record: TemperatureCreate, db: Session = Depends(get_db)):
    crud = TemperatureCRUD(db_session=db)
    return crud.create_record(record)


@router.get("/temperature/range/", response_model=List[TemperatureGet],
            status_code=status.HTTP_200_OK)
def read_interval(start_interval: datetime, end_interval: datetime, db: Session = Depends(get_db)):
    crud = TemperatureCRUD(db_session=db)
    records = crud.read_interval(start_interval, end_interval)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Item not found")
    return records






