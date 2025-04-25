from typing import List

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from classify.base import BaseClassifier
from llm.client import get_llm_client


def build_classification_router(prefix: str, tag: str, classifier_cls: BaseClassifier):
    router = APIRouter(prefix=prefix, tags=[tag])

    class BatchActivityRequest(BaseModel):
        queries: List[str]

    class ClassificationResponse(BaseModel):
        code_ape: str

        class Config:
            json_schema_extra = {"example": {"code_ape": "10.71C"}}

    @router.get(
        "/classify",
        response_model=ClassificationResponse,
        summary="Classify a single activity",
        description="Takes a query string and returns the most appropriate APE code.",
    )
    async def classify_single(request: Request, query: str = Query(...)):
        try:
            async with get_llm_client() as client:
                classifier = classifier_cls(request.app.state.db, client)
                code = await classifier.classify_one(query)
                return {"activity": query, "code_ape": code}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/batch",
        response_model=List[ClassificationResponse],
        summary="Classify a batch of activities",
        description="Takes a list of query strings and returns the most appropriate APE codes for each.",
    )
    async def classify_batch(request: Request, req: BatchActivityRequest):
        try:
            async with get_llm_client() as client:
                classifier = classifier_cls(request.app.state.db, client)
                results = await classifier.classify_batch(req.queries, cancel_check=request.is_disconnected)
                return results
        except HTTPException:
            raise

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
