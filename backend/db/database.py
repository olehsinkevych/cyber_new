import logging
from backend.core.config import DATABASE_URL

logger = logging.getLogger(__name__)


async def check_db_connected():
    try:
        if not str(DATABASE_URL).__contains__("sqlite"):
            database = databases.Database(DATABASE_URL)
            if not database.is_connected:
                database.connect()
                database.execute("SELECT 1")
        logger.info("--- Database is connected (^_^) ---")
        print("Database is connected (^_^)")
    except Exception as e:
        logger.warning("--- DB CONNECTION ERROR ---")
        logger.warning(e)
        logger.warning("--- DB CONNECTION ERROR ---")
        print(
            "Looks like db is missing or is there is some problem in connection,see below traceback"
        )


async def check_db_disconnected():
    try:
        if not str(DATABASE_URL).__contains__("sqlite"):
            database = databases.Database(DATABASE_URL)
            if database.is_connected:
                database.disconnect()
        logger.info("--- Database is disconnected (^_^) ---")
        print("Database is Disconnected (-_-) zZZ")
    except Exception as e:
        logger.warning("--- DB CONNECTION ERROR ---")
        logger.warning(e)
        logger.warning("--- DB CONNECTION ERROR ---")