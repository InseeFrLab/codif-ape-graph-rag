import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, List, Optional

from fastapi import HTTPException
from langchain_neo4j import Neo4jVector
from openai import AsyncOpenAI


class BaseClassifier(ABC):
    def __init__(self, db: Neo4jVector, client: AsyncOpenAI):
        self.db = db
        self.client = client

    @abstractmethod
    async def classify_one(self, query: str) -> str:
        pass

    async def classify_batch(
        self,
        queries: List[str],
        cancel_check: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> List[dict]:
        # This functions allows to stop server side processes if client received a timeout (on veut pas que le LLM continue à tourner alors qu'on a chopper un timeout)
        # On le wrap pour eviter de changer dans les 4 méthodes
        async def wrap_classify_one(q: str) -> str:
            if cancel_check and await cancel_check():
                raise asyncio.CancelledError("Client disconnected")
            return await self.classify_one(q)

        try:
            tasks = [wrap_classify_one(q) for q in queries]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            raise HTTPException(status_code=499, detail="Client disconnected")

        return [{"code_ape": code if not isinstance(code, Exception) else "ERROR"} for act, code in zip(queries, raw_results)]
