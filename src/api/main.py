# src/api/main.py
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from classify import classify_activities_batch, classify_activity
from llm.client import get_llm_client
from utils.logging import configure_logging
from vector_db.loaders import get_vector_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the lifespan of the API.
    This context manager is used to load the ML model and other resources
    when the API starts and clean them up when the API stops.
    Args:
        app (FastAPI): The FastAPI application.
    """
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting API lifespan")
    app.state.db = get_vector_db()
    yield
    logger.info("🛑 Shutting down API lifespan")


app = FastAPI(title="Codif APE Classifier API", version="1.0.0", lifespan=lifespan)


class BatchActivityRequest(BaseModel):
    queries: List[str]


class ClassificationResponse(BaseModel):
    code_ape: str

    class Config:
        json_schema_extra = {"example": {"code_ape": "1071C"}}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify a single activity",
    description="Takes a query string and returns the most appropriate APE code.",
)
async def classify_single(query: str = Query(..., description="Activity to classify")):
    logger = logging.getLogger(__name__)
    logger.info(f"📩 Received classification request: {query}")
    try:
        async with get_llm_client() as client:
            code = await classify_activity(query, app.state.db, client)
            return {"code_ape": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.post(
    "/batch",
    response_model=List[ClassificationResponse],
    summary="Classify a batch of activities",
    description="Takes a list of query strings and returns the most appropriate APE codes for each.",
)
async def classify_batch(req: BatchActivityRequest):
    logger = logging.getLogger(__name__)
    logger.info(f"📦 Received batch of {len(req.queries)} activities")
    try:
        async with get_llm_client() as client:
            results = await classify_activities_batch(req.queries, app.state.db, client)
            return results
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
