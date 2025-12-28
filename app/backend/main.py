from api.v1.router import router as router_v1
from fastapi import FastAPI

app = FastAPI()

app.include_router(router_v1, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Hello World"}
