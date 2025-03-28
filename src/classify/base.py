import asyncio
from abc import ABC, abstractmethod
from typing import List

from langchain_neo4j import Neo4jVector
from openai import AsyncOpenAI


class BaseClassifier(ABC):
    def __init__(self, db: Neo4jVector, client: AsyncOpenAI):
        self.db = db
        self.client = client

    @abstractmethod
    async def classify_one(self, query: str) -> str:
        pass

    async def classify_batch(self, queries: List[str]) -> List[dict]:
        tasks = [self.classify_one(q) for q in queries]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        return [{"code_ape": code if not isinstance(code, Exception) else "ERROR"} for act, code in zip(queries, raw_results)]
