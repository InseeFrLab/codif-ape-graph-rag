import logging

from classify.base import BaseClassifier
from vector_db.utils import is_final_code, retrieve_docs_for_code

logger = logging.getLogger(__name__)


class EmbeddingHierarchicalClassifier(BaseClassifier):
    async def classify_one(self, query: str) -> str:
        try:
            retrieved_docs = await self.db.asimilarity_search(f"query : {query}", k=1, filter={"LEVEL": 1})
            selected_code = retrieved_docs[0].metadata["CODE"]
            logger.info("📌 Niveau 1 : %s", selected_code)

            while not is_final_code(selected_code, retrieved_docs):
                retrieved_docs = await retrieve_docs_for_code(selected_code, query, self.db)
                selected_code = retrieved_docs[0].metadata["CODE"]
                level = retrieved_docs[0].metadata["LEVEL"]
                logger.info("🔍 Niveau %d : %d documents", level, len(retrieved_docs))
                logger.info("📌 Niveau %d : %s", level, selected_code)

            return selected_code

        except Exception as e:
            logger.exception("❌ Erreur classification : %s", e)
            raise
