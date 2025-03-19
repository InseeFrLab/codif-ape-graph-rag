import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph

# Connexion à Neo4j via langchain
graph = Neo4jGraph(
    url="neo4j://neo4j-426097.projet-ape:7687",
    username="neo4j",
    password="**",
    enhanced_schema=True,
)

# Charger votre fichier Excel
df = pd.read_excel("codif-ape-graph-rag/NACE_Rev2.1_Structure_Explanatory_Notes_EN.xlsx", sheet_name="Sheet0")


def concatenate_columns(row):
    cols = ["NAME", "Implementation_rule", "Includes", "IncludesAlso", "Excludes"]
    return "\n\n".join([str(row[col]) for col in cols if pd.notnull(row[col])])


df["text_content"] = df.apply(concatenate_columns, axis=1)

# Affichage pour vérification
df[["NAME", "Implementation_rule", "Includes", "IncludesAlso", "Excludes", "text_content"]].head()

emb_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=False,
)

df["embedding"] = df["text_content"].apply(lambda x: emb_model.embed_query(x))


# Fonction pour importer les données vers Neo4j
def integrate_embeddings(df, graph):
    # Création de la contrainte d'unicité
    graph.query("""
    CREATE CONSTRAINT activity_id_unique IF NOT EXISTS FOR (a:Activity) REQUIRE a.id IS UNIQUE
    """)

    for _, row in df.iterrows():
        embedding_vector = [float(x) for x in row["embedding"]]

        graph.query(
            """
        MERGE (a:Activity {id: $id})
        SET a.code = $code,
            a.name = $name,
            a.level = $level,
            a.text_content = $text_content,
            a.embedding = $embedding
        """,
            params={
                "id": str(row["ID"]),
                "code": row["CODE"],
                "name": row["NAME"],
                "level": int(row["LEVEL"]) if pd.notnull(row["LEVEL"]) else None,
                "text_content": row["text_content"],
                "embedding": embedding_vector,
            },
        )

    # Insertion des relations parent-enfant
    for _, row in df.dropna(subset=["PARENT_ID"]).iterrows():
        graph.query(
            """
        MERGE (child:Activity {id: $child_id})
        MERGE (parent:Activity {id: $parent_id})
        MERGE (child)-[:HAS_PARENT]->(parent)
        """,
            params={"child_id": str(row["ID"]), "parent_id": str(row["PARENT_ID"])},
        )


# Exécutez l'importation
integrate_embeddings(df, graph)
