from sqlalchemy.orm.session import Session
from sqlalchemy.future import select
from sqlalchemy import update
from typing import Optional
from backend.models.measurements import Temperature
from backend.schemas.measurements import TemperatureCreate


class TemperatureCRUD:

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def read_on_id(self, id_: int):
        return self.db_session.query(Temperature).get(id_)

    def create_record(self, measurement: TemperatureCreate):
        db_record = Temperature(time_point=measurement.time_point,
                                user=measurement.user,
                                tag=measurement.tag, value=measurement.value)
        self.db_session.add(db_record)
        self.db_session.commit()
        self.db_session.refresh(db_record)
        return db_record

