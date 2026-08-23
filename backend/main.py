from fastapi import FastAPI

from backend.db import create_all_tables

app = FastAPI(title="Promise Integrity Agent")


@app.on_event("startup")
def on_startup() -> None:
    create_all_tables()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
