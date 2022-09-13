from backend.db.session import session_local, engine
from sqlalchemy.orm.session import Session


def get_db() -> Session:
    db = session_local()
    try:
        yield db
    finally:
        db.close()
