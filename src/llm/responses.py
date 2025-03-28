import asyncio
import logging

from openai import OpenAI

from constants.llm import GENERATION_MODEL
from constants.prompts import SYS_PROMPT
from llm.schema import Response

logger = logging.getLogger(__name__)


async def get_llm_choice(prompt: str, client: OpenAI, retries: int = 3, delay: float = 2.0) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = await client.beta.chat.completions.parse(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=Response,
                extra_body={"guided_decoding_backend": "guidance"},
            )
            return response.choices[0].message.parsed.code

        except Exception as e:
            logger.warning("⚠️ LLM erreur tentative %d : %s", attempt, str(e))
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
