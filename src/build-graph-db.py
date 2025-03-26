import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector

from constants.graph_db import EMBEDDING_MODEL, NEO4J_PWD, NEO4J_URL, NEO4J_USERNAME
from utils.data import get_file_system

fs = get_file_system()

emb_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    # encode_kwargs={"normalize_embeddings": True},
    show_progress=False,
)

# On récupère les notices
df = pd.read_parquet("projet-ape/notices/Notices-NAF2025-FR.parquet", filesystem=fs)

docs = DataFrameLoader(
    df[["ID", "CODE", "NAME", "PARENT_ID", "PARENT_CODE", "LEVEL", "FINAL", "text_content"]], page_content_column="text_content"
).load()

# On créer la base avec les vecteurs d'embeddings
db = Neo4jVector.from_documents(docs, emb_model, url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PWD)

# On se connect à la base créée précedemment
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PWD,
    enhanced_schema=True,
)

# On rajoute les relations
graph.query(
    """
    MATCH (parent), (child)
    WHERE parent.ID = child.PARENT_ID AND child.PARENT_ID IS NOT NULL
    MERGE (parent)-[:HAS_CHILD]->(child)
    """
)


query_text = "query : Je suis boulanger dans le 93"

db.similarity_search_with_relevance_scores(query_text, k=2, filter={"FINAL": 1})
