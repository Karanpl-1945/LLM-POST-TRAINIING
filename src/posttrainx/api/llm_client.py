from __future__ import annotations

import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI


class LLMClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "local-dev-key"),
        )
        self.model = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B")

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=256,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
