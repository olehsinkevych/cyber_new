from fastapi import FastAPI
from backend.api.routes import router as api_router


def app_factory():
    app = FastAPI(title='Demo', version="0.1.0")
    app.include_router(api_router)
    return app


app = app_factory()

'''
@app.on_event("startup")
async def app_startup():
    await check_db_connected()


@app.on_event("shutdown")
async def app_shutdown():
    await check_db_disconnected()
'''

