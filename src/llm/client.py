from contextlib import asynccontextmanager

import httpx
from openai import AsyncOpenAI

from constants.llm import URL_LLM_API


@asynccontextmanager
async def get_llm_client_async():
    client = AsyncOpenAI(api_key="EMPTY", base_url=URL_LLM_API, timeout=httpx.Timeout(30.0))
    try:
        yield client
    finally:
        await client.close()
