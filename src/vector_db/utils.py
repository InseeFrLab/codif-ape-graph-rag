import asyncio
import logging
from typing import List

from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_neo4j import Neo4jVector

from utils.cypher import count_children

logger = logging.getLogger(__name__)


def truncate_docs_to_max_tokens(docs, max_tokens):
    splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    truncated_docs = []

    for doc in docs:
        original_text = doc.page_content
        chunks = splitter.split_text(original_text)

        if len(chunks) > 1:
            logger.warning(f"Document truncated to {max_tokens} tokens. Metadata: {doc.metadata}")

        doc.page_content = chunks[0]
        truncated_docs.append(doc)

    return truncated_docs


def dicts_to_documents(results: List[dict]) -> List[Document]:
    docs = []
    for record in results:
        node = record["n"]
        metadata_keys = ["FINAL", "NAME", "PARENT_CODE", "ID", "LEVEL", "PARENT_ID", "CODE"]
        metadata = {key: node[key] for key in metadata_keys if key in node}
        content = f"\ntext: {node['text']}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


async def retrieve_docs_for_code(code: str, query_text: str, db: Neo4jVector) -> List[Document]:
    """
    Async version to retrieve APE code child documents, using similarity search or direct query.
    """
    if await count_children(db, code) > 5:
        return await db.asimilarity_search(f"query : {query_text}", k=5, filter={"PARENT_CODE": code})
    else:
        raw = await asyncio.to_thread(
            db.query,
            f"""
            MATCH (n)
            WHERE n.PARENT_CODE = '{code}'
            RETURN n
        """,
        )
        return dicts_to_documents(raw)


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
