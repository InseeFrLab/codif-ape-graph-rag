import asyncio
import logging

from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)


def create_parent_child_relationships(graph: Neo4jGraph):
    logger.info("🔁 Creating HAS_CHILD relationships")
    graph.query(
        """
    MATCH (child)
    WHERE child.PARENT_ID IS NOT NULL
    MATCH (parent {ID: child.PARENT_ID})
    MERGE (parent)-[:HAS_CHILD]->(child)
    """
    )
    logger.info("✅ Relationships created")


def count_children(graph: Neo4jGraph, code: str) -> int:
    return asyncio.run(count_children_async())


async def count_children_async(graph: Neo4jGraph, code: str) -> int:
    query = f"""
    MATCH (n)
    WHERE n.PARENT_CODE = '{code}'
    RETURN COUNT(n) AS count
    """
    result = await asyncio.to_thread(graph.query, query)
    return result[0]["count"] if result else 0
