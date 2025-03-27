from typing import List

from langchain.schema import Document

from constants.prompts import CLASSIF_PROMPT


def format_prompt(activity: str, docs: List[Document]) -> str:
    codes = "\n\n".join(f"##########\nCode APE : {doc.metadata['CODE']}{doc.page_content}" for doc in docs)
    list_codes = ", ".join(f"'{doc.metadata['CODE']}'" for doc in docs)
    return CLASSIF_PROMPT.format(activity=activity, proposed_codes=codes, list_proposed_codes=list_codes)
