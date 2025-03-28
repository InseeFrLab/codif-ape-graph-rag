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

    @abstractmethod
    async def classify_batch(self, queries: List[str]) -> List[dict]:
        pass
