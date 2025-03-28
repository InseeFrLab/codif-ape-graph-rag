import asyncio
import logging
from typing import List

from classify.base import BaseClassifier

logger = logging.getLogger(__name__)


class EmbeddingFlatClassifier(BaseClassifier):
    async def classify_one(self, query: str) -> str:
        try:
            retrieved_docs = await self.db.asimilarity_search(f"query : {query}", k=1, filter={"FINAL": 1})
            selected_code = retrieved_docs[0].metadata["CODE"]
            logger.info("📌 Niveau 5 : %s", selected_code)
            return selected_code

        except Exception as e:
            logger.exception("❌ Erreur classification : %s", e)
            return "ERROR"

    async def classify_batch(self, queries: List[str]) -> List[dict]:
        tasks = [self.classify_one(q) for q in queries]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        return [{"code_ape": code if not isinstance(code, Exception) else "ERROR"} for act, code in zip(queries, raw_results)]
