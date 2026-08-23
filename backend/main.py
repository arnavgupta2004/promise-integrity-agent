from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from backend.db import create_all_tables
from backend.routes.dashboard import router as dashboard_router

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Promise Integrity Agent")
app.include_router(dashboard_router)


@app.on_event("startup")
def on_startup() -> None:
    create_all_tables()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
