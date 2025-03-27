from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector

from constants.graph_db import EMBEDDING_MODEL, NEO4J_PWD, NEO4J_URL, NEO4J_USERNAME


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Initialize the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


def get_vector_db() -> Neo4jVector:
    """Initialize the Neo4jVector Store from existing graph."""
    emb_model = get_embedding_model(EMBEDDING_MODEL)
    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PWD)
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
