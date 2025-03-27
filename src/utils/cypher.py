from langchain_neo4j import Neo4jGraph


def count_children(graph: Neo4jGraph, code: str) -> int:
    query = f"""
    MATCH (n)
    WHERE n.PARENT_CODE = '{code}'
    RETURN COUNT(n) AS count
    """
    result = graph.query(query)
    return result[0]["count"] if result else 0
