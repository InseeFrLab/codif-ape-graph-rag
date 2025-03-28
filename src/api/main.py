# src/api/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import flat_embeddings, flat_rag, hierarchical_embeddings, hierarchical_rag
from utils.logging import configure_logging
from vector_db.loaders import get_vector_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the lifespan of the API.
    Loads the vector DB once at startup and attaches it to app state.
    """
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting API lifespan")
    app.state.db = get_vector_db()
    yield
    logger.info("🛑 Shutting down API lifespan")


app = FastAPI(title="Codif APE Classifier API", version="1.0.0", lifespan=lifespan)
app.include_router(flat_embeddings.router)
app.include_router(hierarchical_embeddings.router)
app.include_router(flat_rag.router)
app.include_router(hierarchical_rag.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
