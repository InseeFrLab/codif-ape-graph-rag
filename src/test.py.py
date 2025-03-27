import logging
import time
from typing import List

from langchain_neo4j import Neo4jVector
from openai import OpenAI

from constants.llm import URL_LLM_API
from llm.prompting import format_prompt
from llm.responses import get_llm_choice
from vector_db.loaders import get_vector_db
from vector_db.utils import is_final_code, retrieve_docs_for_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


def classify_activity(query_text: str, db: Neo4jVector, client: OpenAI) -> str:
    """
    Classifies a given activity description through the APE code hierarchy using a vector DB and LLM.

    Args:
        query_text (str): The activity description to classify.
        db (Neo4jVector): The connected vector DB object.
        client (OpenAI):
    Returns:
        str: The final selected APE code.
    """
    start_time = time.time()

    try:
        logger.info("🔍 Starting classification for query: %s", query_text)

        # Niveau 1
        retrieved_docs = db.similarity_search(f"query : {query_text}", k=5, filter={"LEVEL": 1})
        prompt = format_prompt(query_text, retrieved_docs)
        selected_code = get_llm_choice(prompt, client)
        logger.info("📌 Niveau 1 - Code sélectionné: %s", selected_code)

        # Deep classification loop
        for level in range(2, 6):
            retrieved_docs = retrieve_docs_for_code(selected_code, query_text, db)
            prompt = format_prompt(query_text, retrieved_docs)
            selected_code = get_llm_choice(prompt, client)
            logger.info("📌 Niveau %d - Code sélectionné: %s", level, selected_code)

            if is_final_code(selected_code, retrieved_docs):
                logger.info("✅ Code FINAL atteint à niveau %d: %s", level, selected_code)
                break

        elapsed = time.time() - start_time
        logger.info("🎯 Classification terminée en %.2f secondes", elapsed)
        return selected_code

    except Exception as e:
        logger.exception("❌ Erreur pendant la classification: %s", str(e))
        raise


def _classify_activity(query_text: str, db: Neo4jVector, client: OpenAI) -> str:
    """
    Classifies a given activity description through the APE code hierarchy using a vector DB and LLM.

    Args:
        query_text (str): The activity description to classify.
        db (Neo4jVector): The connected vector DB object.
        client (OpenAI):
    Returns:
        str: The final selected APE code.
    """
    try:
        # Niveau 1
        retrieved_docs = db.similarity_search(f"query : {query_text}", k=5, filter={"LEVEL": 1})
        prompt = format_prompt(query_text, retrieved_docs)
        selected_code = get_llm_choice(prompt, client)

        # Deep classification loop
        for level in range(2, 6):
            retrieved_docs = retrieve_docs_for_code(selected_code, query_text, db)
            prompt = format_prompt(query_text, retrieved_docs)
            selected_code = get_llm_choice(prompt, client)

            if is_final_code(selected_code, retrieved_docs):
                break

        return selected_code

    except Exception as e:
        logger.exception("❌ Erreur pendant la classification: %s", str(e))
        raise


def classify_activities_batch(activities: List[str], db: Neo4jVector, client: OpenAI):
    results = []

    logger.info("🚀 Lancement du traitement batch de %d activités", len(activities))
    start = time.time()

    for i, activity in enumerate(activities, 1):
        logger.info("⚙️ Traitement %d/%d", i, len(activities))
        code = _classify_activity(activity, db, client)
        results.append({"activity": activity, "code_ape": code})

    elapsed = time.time() - start
    logger.info("✅ Batch terminé en %.2f secondes", elapsed)
    return results


if __name__ == "__main__":
    db = get_vector_db()
    client = OpenAI(api_key="EMPTY", base_url=URL_LLM_API)
    query_text = "loueur de meublé non professionnel"
    final_code = classify_activity(query_text, db)
    logger.info("🏁 Code APE final sélectionné : %s", final_code)

    activities_to_classify = [
        "Je suis boulanger dans le 93",
        "Fabrication de meubles design en bois recyclé",
        "Conseiller en patrimoine indépendant",
        "Transport de marchandises dangereuses",
        "Réparation de trottinettes électriques",
    ]
    classify_activities_batch(activities_to_classify, db, client)
