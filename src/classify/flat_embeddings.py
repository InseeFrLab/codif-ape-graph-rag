import logging

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
