import logging

from classify.base import BaseClassifier
from llm.prompting import format_prompt
from llm.responses import get_llm_choice
from vector_db.utils import is_final_code, retrieve_docs_for_code

logger = logging.getLogger(__name__)


class RAGHierarchicalClassifier(BaseClassifier):
    async def classify_one(self, query: str) -> str:
        try:
            retrieved_docs = await self.db.asimilarity_search(f"query : {query}", k=5, filter={"LEVEL": 1})
            prompt = format_prompt(query, retrieved_docs)
            selected_code = await get_llm_choice(prompt, self.client)
            logger.info("📌 Niveau 1 : %s", selected_code)

            for level in range(2, 6):
                retrieved_docs = await retrieve_docs_for_code(selected_code, query, self.db)
                prompt = format_prompt(query, retrieved_docs)
                selected_code = await get_llm_choice(prompt, self.client)
                logger.info("📌 Niveau %d : %s", level, selected_code)

                if is_final_code(selected_code, retrieved_docs):
                    break

            return selected_code

        except Exception as e:
            logger.exception("❌ Erreur classification : %s", e)
            raise
