from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, Float
import pandas as pd

from backend.db.session import engine


def read_save_base(path='backend/data/interior_temp.csv') -> None:
    df = pd.read_csv(path)
    df = df.rename(columns={"Unnamed: 0": "id", "sensorId": "tag"})
    df.drop(['id'], axis=1)
    sqlite_connection = engine.connect()
    sqlite_table = "base_temperature"
    meta = MetaData()
    temperature = Table(
        sqlite_table, meta,
        Column('id', Integer, primary_key=True),
        Column('tag', String),
        Column('time_point', DateTime),
        Column('value', Float),
        Column('user', String)
    )
    meta.create_all(sqlite_connection)
    df.to_sql(sqlite_table, sqlite_connection, if_exists='append',
              index=False, index_label=None, chunksize=None)


read_save_base()
