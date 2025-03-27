from langchain_neo4j import Neo4jVector
from openai import OpenAI

from constants.llm import URL_LLM_API
from llm.prompting import format_prompt
from llm.responses import get_llm_choice
from vector_db.loaders import get_vector_db
from vector_db.utils import is_final_code, retrieve_docs_for_code


def classify_activity(query_text: str, db: Neo4jVector) -> str:
    client = OpenAI(api_key="EMPTY", base_url=URL_LLM_API)

    # Initial level - top-level proposals
    retrieved_docs = db.similarity_search(query_text, k=5, filter={"LEVEL": 1})
    prompt = format_prompt(query_text, retrieved_docs)
    selected_code = get_llm_choice(prompt, client)
    print(f"Niveau 1 : {selected_code}")

    # Drill-down through levels
    for level in range(2, 6):
        retrieved_docs = retrieve_docs_for_code(selected_code, query_text, db)
        prompt = format_prompt(query_text, retrieved_docs)
        selected_code = get_llm_choice(prompt, client)
        print(f"Niveau {level} : {selected_code}")
        if is_final_code(selected_code, retrieved_docs):
            print("Le code correspond à la fin de la hiérarchie")
            break

    return selected_code


if __name__ == "__main__":
    db = get_vector_db()
    query_text = "query : loueur de meublé non professionnel"
    final_code = classify_activity(query_text, db)
    print(f"\n✅ Code APE final sélectionné : {final_code}")
