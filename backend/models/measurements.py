from sqlalchemy import Column, Float, String, DateTime, Integer
from backend.db.base_class import Base


class Temperature(Base):
    __tablename__ = "base_temperature"
    id = Column(Integer, primary_key=True, index=True)
    time_point = Column(DateTime, unique=False, index=True, nullable=False)
    user = Column(String(50))
    value = Column(Float, nullable=False)
    tag = Column(String(40))