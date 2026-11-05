# Architecture

## Training path

```text
Raw conversational corpus
  -> PySpark normalization / MinHash-LSH near-dedup
  -> conversational records
  -> Qwen chat template + tokenizer (TRL)
  -> SFT: LoRA vs QLoRA
  -> controlled evaluation
  -> merge selected SFT adapter to BF16 checkpoint
  -> preference pairs (prompt / chosen / rejected)
  -> DPO: LoRA vs QLoRA
  -> merge selected DPO adapter
  -> final aligned model
```

## Online serving path

```text
Client
 -> Nginx / API gateway responsibilities
    - TLS (add in production)
    - rate limiting
    - connection limiting
    - reverse proxy / load balancing
 -> FastAPI product-chat service
    - request validation
    - session/retrieval integration point
    - product catalog retrieval
    - grounding prompt builder
 -> vLLM OpenAI-compatible API
    - scheduler / continuous batching
    - normal KV caching
    - automatic prefix caching
    - paged KV-cache management
    - configurable FP8 KV cache
 -> streamed response
```

## Production changes

The included product retriever is intentionally local and tiny. Replace it with the actual product search/catalog/inventory services. Do not use the LLM as the source of truth for prices, stock, delivery dates, or discounts.

The Kubernetes manifests are reference manifests, not a complete cloud deployment. In production add TLS, secrets management, observability, custom GPU autoscaling, pod disruption budgets, multi-zone placement, model artifact distribution, and authentication/authorization.
