from __future__ import annotations

import json
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from posttrainx.api.catalog import ProductCatalog
from posttrainx.api.llm_client import LLMClient
from posttrainx.api.rate_limit import RedisFixedWindowLimiter

app = FastAPI(title="PostTrainX Product Chat API", version="0.1.0")
catalog = ProductCatalog(os.getenv("PRODUCT_CATALOG_PATH", "data/sample/products.json"))
llm = LLMClient()
limiter = RedisFixedWindowLimiter(requests_per_minute=int(os.getenv("APP_RPM", "60")))

SYSTEM_PROMPT = """You are a shopping assistant. Use only the supplied PRODUCT DATA for product facts.
Never invent price, stock, size availability, discount, or delivery information. If the requested fact is
missing, say that it is unavailable and suggest using the product service to verify it. Be concise."""


class ProductChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation: list[dict[str, str]] = Field(default_factory=list)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/product-chat")
async def product_chat(req: ProductChatRequest, x_user_id: str | None = Header(default=None)):
    identity = x_user_id or "anonymous"
    if not await limiter.allow(identity):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    products = catalog.search(req.query)
    context = json.dumps(products, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *req.conversation[-8:],
        {
            "role": "user",
            "content": f"USER QUERY:\n{req.query}\n\nPRODUCT DATA:\n{context}",
        },
    ]

    async def generate():
        async for token in llm.stream_chat(messages):
            yield token

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
