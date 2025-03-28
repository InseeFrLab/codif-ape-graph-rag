import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from classify.hierarchical_rag import RAGHierarchicalClassifier
from llm.client import get_llm_client

router = APIRouter(prefix="/hierarchical-rag", tags=["Hierarchical RAG"])


class BatchActivityRequest(BaseModel):
    queries: List[str]


class ClassificationResponse(BaseModel):
    code_ape: str

    class Config:
        json_schema_extra = {"example": {"code_ape": "1071C"}}


@router.get(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify a single activity",
    description="Takes a query string and returns the most appropriate APE code.",
)
async def classify_single(request: Request, query: str = Query(..., description="Activity to classify")):
    logger = logging.getLogger(__name__)
    logger.info(f"📩 [RAG Hierarchical] classify_one: {query}")
    try:
        async with get_llm_client() as client:
            classifier = RAGHierarchicalClassifier(request.app.state.db, client)
            code = await classifier.classify_one(query)
            return {"code_ape": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/batch",
    response_model=List[ClassificationResponse],
    summary="Classify a batch of activities",
    description="Takes a list of query strings and returns the most appropriate APE codes for each.",
)
async def classify_batch(request: Request, req: BatchActivityRequest):
    logger = logging.getLogger(__name__)
    logger.info(f"📦 [RAG Hierarchical] classify_batch: {len(req.queries)} queries")
    try:
        async with get_llm_client() as client:
            classifier = RAGHierarchicalClassifier(request.app.state.db, client)
            results = await classifier.classify_batch(req.queries)
            return results
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
