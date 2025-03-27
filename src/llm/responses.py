from openai import OpenAI
from pydantic import BaseModel

from constants.llm import GENERATION_MODEL
from constants.prompts import SYS_PROMPT


class Response(BaseModel):
    code: str


def get_llm_choice(prompt: str, client: OpenAI) -> str:
    response = client.beta.chat.completions.parse(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=Response,
        extra_body={"guided_decoding_backend": "guidance"},
    )
    return response.choices[0].message.parsed.code
