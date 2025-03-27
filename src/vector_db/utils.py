from typing import List

from langchain.schema import Document
from langchain_neo4j import Neo4jVector

from utils.cypher import count_children


def dicts_to_documents(results: List[dict]) -> List[Document]:
    docs = []
    for record in results:
        node = record["n"]
        metadata_keys = ["FINAL", "NAME", "PARENT_CODE", "ID", "LEVEL", "PARENT_ID", "CODE"]
        metadata = {key: node[key] for key in metadata_keys if key in node}
        content = f"\ntext: {node['text']}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def retrieve_docs_for_code(code: str, query_text: str, db: Neo4jVector) -> List[Document]:
    if count_children(db, code) > 5:
        return db.similarity_search(f"query : {query_text}", k=5, filter={"PARENT_CODE": code})
    else:
        raw_results = db.query(f"""
            MATCH (n)
            WHERE n.PARENT_CODE = '{code}'
            RETURN n
        """)
        return dicts_to_documents(raw_results)


def is_final_code(code: str, documents: List[Document]) -> bool:
    """
    Check whether a given APE code is marked as FINAL in a list of documents.
    Args:
        code (str): The code to check.
        documents (List[Document]): A list of Document instances with metadata including 'CODE' and 'FINAL'.

    Returns:
        bool: True if the code is FINAL (i.e., metadata['FINAL'] == 1), False otherwise.
    """
    match = next((doc for doc in documents if doc.metadata.get("CODE") == code), None)
    return bool(match and match.metadata.get("FINAL") == 1)
