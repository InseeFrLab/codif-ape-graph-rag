import logging

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings

# from vector_db.openai_embeddings import CustomOpenAIEmbeddings
from constants.graph_db import EMBEDDING_MODEL, NEO4J_PWD, NEO4J_URL, NEO4J_USERNAME, URL_EMBEDDING_API

logger = logging.getLogger(__name__)


def create_vector_db(docs, embedding_model) -> Neo4jVector:
    logger.info("🧠 Creating Neo4j vector DB with embeddings")
    return Neo4jVector.from_documents(
        docs, embedding_model, url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PWD, ids=[f"{i}" for i in range(len(docs))]
    )


def setup_graph() -> Neo4jGraph:
    logger.info("🔗 Connecting to Neo4j graph DB")
    return Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PWD,
        enhanced_schema=True,
    )


def get_embedding_model(model_name: str) -> OpenAIEmbeddings:
    """Initialize the embedding model."""
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_base=URL_EMBEDDING_API,
        openai_api_key="EMPTY",
        tiktoken_enabled=False,
    )


async def get_vector_db() -> Neo4jVector:
    """Initialize the Neo4jVector Store from existing graph."""
    emb_model = get_embedding_model(EMBEDDING_MODEL)
    graph = setup_graph()
    return Neo4jVector.from_existing_graph(
        graph=graph,
        embedding=emb_model,
        index_name="id",
        node_label="Chunk",
        text_node_properties=["text"],
        keyword_index_name="text",
        embedding_node_property="embedding",
        search_type="vector",
    )
