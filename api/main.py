"""FastAPI app – run with: uvicorn api.main:app --port 8000 --reload"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes import router
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the RAG engine on startup so the first request is fast."""
    logger.info("Starting up …")
    try:
        from api.routes import _get_engine
        _get_engine()
        logger.success("Engine ready.")
    except FileNotFoundError:
        logger.warning("Vector index missing – run build_index.py before serving.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="RAG-based SHL assessment recommender powered by Gemini + FAISS.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} {response.status_code} [{(time.perf_counter()-start)*1000:.0f}ms]")
    return response


app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": settings.API_TITLE, "version": settings.API_VERSION, "docs": "/docs"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
