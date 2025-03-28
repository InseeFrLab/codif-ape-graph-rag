import asyncio
import logging
from typing import List

from classify.base import BaseClassifier
from llm.prompting import format_prompt
from llm.responses import get_llm_choice

logger = logging.getLogger(__name__)


class RAGFlatClassifier(BaseClassifier):
    async def classify_one(self, query: str) -> str:
        try:
            retrieved_docs = await self.db.asimilarity_search(f"query : {query}", k=5, filter={"FINAL": 1})
            prompt = format_prompt(query, retrieved_docs)
            selected_code = await get_llm_choice(prompt, self.client)
            logger.info("📌 Niveau 5 : %s", selected_code)
            return selected_code

        except Exception as e:
            logger.exception("❌ Erreur classification : %s", e)
            return "ERROR"

    async def classify_batch(self, queries: List[str]) -> List[dict]:
        tasks = [self.classify_one(q) for q in queries]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        return [{"code_ape": code if not isinstance(code, Exception) else "ERROR"} for act, code in zip(queries, raw_results)]
