import asyncio
import logging
from typing import List

from langchain_neo4j import Neo4jVector
from openai import AsyncOpenAI

from llm.prompting import format_prompt
from llm.responses import get_llm_choice_async
from vector_db.utils import is_final_code, retrieve_docs_for_code_async

logger = logging.getLogger(__name__)


async def classify_activity_async(query_text: str, db: Neo4jVector, client: AsyncOpenAI) -> str:
    try:
        retrieved_docs = await db.asimilarity_search(f"query : {query_text}", k=5, filter={"LEVEL": 1})
        prompt = format_prompt(query_text, retrieved_docs)
        selected_code = await get_llm_choice_async(prompt, client)
        logger.info("📌 Niveau 1 : %s", selected_code)

        for level in range(2, 6):
            retrieved_docs = await retrieve_docs_for_code_async(selected_code, query_text, db)
            prompt = format_prompt(query_text, retrieved_docs)
            selected_code = await get_llm_choice_async(prompt, client)
            logger.info("📌 Niveau %d : %s", level, selected_code)

            if is_final_code(selected_code, retrieved_docs):
                break

        return selected_code

    except Exception as e:
        logger.exception("❌ Erreur classification : %s", e)
        return "ERROR"


async def classify_activities_batch_async(activities: List[str], db, client) -> List[dict]:
    tasks = [classify_activity_async(a, db, client) for a in activities]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        {"activity": act, "code_ape": code if not isinstance(code, Exception) else "ERROR"}
        for act, code in zip(activities, raw_results)
    ]
