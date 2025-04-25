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
        tasks = [asyncio.create_task(self.classify_one(q)) for q in queries]

        try:
            # This allows to stop server side processes if client received a timeout
            # (on veut pas que le LLM continue à tourner alors qu'on a choppé un timeout)
            while not all(task.done() for task in tasks):
                if cancel_check and await cancel_check():
                    for task in tasks:
                        task.cancel()
                    raise asyncio.CancelledError("Client disconnected")
                await asyncio.sleep(0.2)

            results = []
            for query, task in zip(queries, tasks):
                try:
                    code = await task
                except asyncio.CancelledError:
                    code = "CANCELLED"
                except Exception:
                    code = "ERROR"
                results.append({"code_ape": code})
            return results

        except asyncio.CancelledError:
            raise HTTPException(status_code=499, detail="Client disconnected")
