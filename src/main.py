import asyncio
import logging

from classify import classify_activities_batch, classify_activity
from llm.client import get_llm_client
from utils.logging import configure_logging
from vector_db.loaders import get_vector_db

configure_logging()
logger = logging.getLogger(__name__)

activities = [
    "Boulanger dans le 93",
    "Vente de voitures d’occasion",
    "Consultant cybersécurité freelance",
] * 10


async def main():
    db = get_vector_db()
    async with get_llm_client() as client:
        results = await classify_activities_batch(activities, db, client)
        for res in results:
            logger.info("✅ %s : %s", res["activity"], res["code_ape"])


async def classify_one():
    db = get_vector_db()
    async with get_llm_client() as client:
        query = "Je suis loueur de meublés non professionnel"
        code = await classify_activity(query, db, client)
        logger.info("✅ Code APE : %s", code)


if __name__ == "__main__":
    asyncio.run(classify_one())
