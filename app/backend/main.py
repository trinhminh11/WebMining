from .api.v1.router import router as router_v1
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router_v1, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Hello World"}
