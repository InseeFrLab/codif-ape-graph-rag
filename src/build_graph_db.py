import logging

from langchain_community.document_loaders import DataFrameLoader

from constants.graph_db import EMBEDDING_MODEL
from constants.paths import NOTICES_PATH
from utils.cypher import create_parent_child_relationships
from utils.data import load_notices
from utils.logging import configure_logging
from vector_db.loaders import create_vector_db, get_embedding_model, setup_graph

configure_logging()

logger = logging.getLogger(__name__)

COLUMNS_TO_KEEP = ["ID", "CODE", "NAME", "PARENT_ID", "PARENT_CODE", "LEVEL", "FINAL", "text_content"]


def run_pipeline():
    df = load_notices(NOTICES_PATH, COLUMNS_TO_KEEP)

    docs = DataFrameLoader(df, page_content_column="text_content").load()

    emb_model = get_embedding_model(EMBEDDING_MODEL)
    _ = create_vector_db(docs, emb_model)

    graph = setup_graph()
    create_parent_child_relationships(graph)


if __name__ == "__main__":
    run_pipeline()
