from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from openai import OpenAI
from pydantic import BaseModel

from constants.graph_db import EMBEDDING_MODEL, NEO4J_PWD, NEO4J_URL, NEO4J_USERNAME

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PWD)

emb_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    # encode_kwargs={"normalize_embeddings": True},
    show_progress=False,
)

db = Neo4jVector.from_existing_graph(
    graph=graph,
    embedding=emb_model,
    index_name="id",
    node_label="Chunk",
    text_node_properties=["text"],  # Propriété pour enrichir le résultat
    keyword_index_name="text",
    embedding_node_property="embedding",
    search_type="vector",
)


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_base = "https://projet-ape-vllm.user.lab.sspcloud.fr/v1"
model = "mistralai/Mistral-Small-24B-Instruct-2501"

client = OpenAI(
    api_key="EMPTY",
    base_url=openai_api_base,
)


###########
query_text = "query : Je suis boulanger dans le 93"

###### NIVEAU 1
# Je chope 5 propals pour les niveaux 1
retrieved_docs = db.similarity_search(query_text, k=5, filter={"LEVEL": 1})

# Je demande au LLM lequel il choisi
SYS_PROMPT = """Tu es un expert de la nomenclature APE. A chaque niveau, choisis le code le plus pertinent pour classifier la description fournie. Tu peux revenir en arrière si besoin ou dire que l'activité est inclassable."""
CLASSIF_PROMPT = """\
* L'activité principale de l'entreprise est : {activity}\n

* Voici la liste des codes APE potentiels et leurs notes explicatives :\n
{proposed_codes}\n
##########
* Le résultat doit être formatté comme une instance JSON qui est conforme au schema ci-dessous. Voici un exemple de résultat à retourner :\n```json\n{{"properties": {{"code": {{"description": "Le code APE sélectionné", "title": "code", "type": "string"}}}}, "required": ["code"]}}\n```

* Le code sélectionné doit absolument faire partie de cette liste des codes APE suivant : [{list_proposed_codes}].

"""


class Response(BaseModel):
    code: str


prompt = CLASSIF_PROMPT.format(
    activity=query_text,
    proposed_codes="\n\n".join([f"##########\nCode APE : {doc.metadata['CODE']}{doc.page_content}" for doc in retrieved_docs]),
    list_proposed_codes=", ".join([f"'{doc.metadata['CODE']}'" for doc in retrieved_docs]),
)


# # Besoin d'un import langchain relou
# parser = PydanticOutputParser(pydantic_object=Response)
# chat_response = client.chat.completions.create(
#     model=model,
#     messages=[
#         {"role": "system", "content": SYS_PROMPT},
#         {"role": "user", "content": prompt},
#     ],
# )
# selected_code = parser.parse(chat_response.choices[0].message.content).code


# chat_response_guided_json = client.chat.completions.create(
#     model=model,
#     messages=[
#         {"role": "system", "content": SYS_PROMPT},
#         {"role": "user", "content": prompt},
#     ],
#     extra_body={"guided_json": Response.model_json_schema()},
# )
# selected_code = Response.model_validate_json(chat_response_guided_json.choices[0].message.content).code


parsed_response = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ],
    response_format=Response,
    extra_body=dict(guided_decoding_backend="guidance"),
)
selected_code = parsed_response.choices[0].message.parsed.code

###### NIVEAU 2


def count_children(graph, code):
    query = f"""
    MATCH (n)
    WHERE n.PARENT_CODE = '{code}'
    RETURN COUNT(n) AS count
    """
    result = graph.query(query)
    return result[0]["count"] if result else 0


def dicts_to_documents(results):
    docs = []
    for record in results:
        node = record["n"]
        metadata_to_keep = ["FINAL", "NAME", "PARENT_CODE", "ID", "LEVEL", "PARENT_ID", "CODE"]
        metadata = {key: node[key] for key in metadata_to_keep if key in node}
        content = f"\ntext: {node['text']}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


if count_children(graph, selected_code) > 5:
    # On fait similarity search
    retrieved_docs = db.similarity_search(query_text, k=5, filter={"PARENT_CODE": selected_code})
else:
    raw_results = db.query(f"""
        MATCH (n)
        WHERE n.PARENT_CODE = '{selected_code}'
        RETURN n
    """)
    retrieved_docs = dicts_to_documents(raw_results)


prompt = CLASSIF_PROMPT.format(
    activity=query_text,
    proposed_codes="\n\n".join([f"##########\nCode APE : {doc.metadata['CODE']}{doc.page_content}" for doc in retrieved_docs]),
    list_proposed_codes=", ".join([f"'{doc.metadata['CODE']}'" for doc in retrieved_docs]),
)

parsed_response = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ],
    response_format=Response,
    extra_body=dict(guided_decoding_backend="guidance"),
)
selected_code = parsed_response.choices[0].message.parsed.code

###### NIVEAU 3

if count_children(graph, selected_code) > 5:
    # On fait similarity search
    retrieved_docs = db.similarity_search(query_text, k=5, filter={"PARENT_CODE": selected_code})
else:
    raw_results = db.query(f"""
        MATCH (n)
        WHERE n.PARENT_CODE = '{selected_code}'
        RETURN n
    """)
    retrieved_docs = dicts_to_documents(raw_results)

prompt = CLASSIF_PROMPT.format(
    activity=query_text,
    proposed_codes="\n\n".join([f"##########\nCode APE : {doc.metadata['CODE']}{doc.page_content}" for doc in retrieved_docs]),
    list_proposed_codes=", ".join([f"'{doc.metadata['CODE']}'" for doc in retrieved_docs]),
)

parsed_response = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ],
    response_format=Response,
    extra_body=dict(guided_decoding_backend="guidance"),
)
selected_code = parsed_response.choices[0].message.parsed.code
print(selected_code)

###### NIVEAU 4


if count_children(graph, selected_code) > 5:
    # On fait similarity search
    retrieved_docs = db.similarity_search(query_text, k=5, filter={"PARENT_CODE": selected_code})
else:
    raw_results = db.query(f"""
        MATCH (n)
        WHERE n.PARENT_CODE = '{selected_code}'
        RETURN n
    """)
    retrieved_docs = dicts_to_documents(raw_results)

prompt = CLASSIF_PROMPT.format(
    activity=query_text,
    proposed_codes="\n\n".join([f"##########\nCode APE : {doc.metadata['CODE']}{doc.page_content}" for doc in retrieved_docs]),
    list_proposed_codes=", ".join([f"'{doc.metadata['CODE']}'" for doc in retrieved_docs]),
)

parsed_response = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ],
    response_format=Response,
    extra_body=dict(guided_decoding_backend="guidance"),
)
selected_code = parsed_response.choices[0].message.parsed.code

###### NIVEAU 5

if count_children(graph, selected_code) > 5:
    # On fait similarity search
    retrieved_docs = db.similarity_search(query_text, k=5, filter={"PARENT_CODE": selected_code})
else:
    raw_results = db.query(f"""
        MATCH (n)
        WHERE n.PARENT_CODE = '{selected_code}'
        RETURN n
    """)
    retrieved_docs = dicts_to_documents(raw_results)

prompt = CLASSIF_PROMPT.format(
    activity=query_text,
    proposed_codes="\n\n".join([f"##########\nCode APE : {doc.metadata['CODE']}{doc.page_content}" for doc in retrieved_docs]),
    list_proposed_codes=", ".join([f"'{doc.metadata['CODE']}'" for doc in retrieved_docs]),
)


parsed_response = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ],
    response_format=Response,
    extra_body=dict(guided_decoding_backend="guidance"),
)
selected_code = parsed_response.choices[0].message.parsed.code
