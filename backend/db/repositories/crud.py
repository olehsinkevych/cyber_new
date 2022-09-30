from datetime import datetime
from sqlalchemy.orm.session import Session
from sqlalchemy.future import select
from sqlalchemy import update
from sqlalchemy import and_
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

    def read_interval(self, start_date: datetime, end_date: datetime):
        return self.db_session.query(Temperature).filter(and_(
            Temperature.time_point >= start_date,
            Temperature.time_point <= end_date)).all()

    def update_temperature(self):
        pass

    def delete_temperature(self):
        pass

    def read_all(self):
        pass
