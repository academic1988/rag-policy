# Design & Evaluation

## 1. Architecture & Design Decisions

### Document Corpus
Ten company policy documents were assembled covering PTO, remote work, expenses, code of conduct, information security, holidays, performance reviews, leave of absence, data privacy, and learning & development — totalling ~120 policy paragraphs. Documents are stored as `.txt` (easily replaceable with PDFs).

### Chunking Strategy
| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 512 characters | Balances context density with retrieval precision; policy clauses typically fit in ~1–3 sentences |
| Overlap | 64 characters | Prevents splitting key facts across chunk boundaries |
| Splitter | `RecursiveCharacterTextSplitter` | Respects paragraph/sentence boundaries before splitting mid-sentence |

A character-based splitter was chosen over a token-based one for simplicity and zero external tokeniser dependency at index time. Overlap of ~12% ensures no clause is stranded at a boundary.

### Embedding Model
**`all-MiniLM-L6-v2`** (SentenceTransformers, 22M params, 384-dim)
- Runs locally — no API cost, no rate limits during indexing
- Achieves strong semantic similarity for English policy text
- Fast: ~1,000 sentences/second on CPU
- Cosine similarity stored in ChromaDB with HNSW index for sub-millisecond retrieval at this corpus size

### Vector Store
**ChromaDB (persistent local)**
- Zero infrastructure — single file on disk, ideal for free-tier hosting
- HNSW index gives O(log n) approximate nearest-neighbour search
- For larger corpora (>1M chunks) or multi-tenant production, Pinecone or Weaviate would be preferred

### Retrieval
- **Top-k = 5** — empirically balances recall (enough context for multi-clause policies) against prompt token budget (~2,500 chars of context per query)
- Optional reranking step is a no-op (identity sort by cosine score) but the hook is in place to swap in a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) if latency budget allows

### LLM
**`openrouter/free`** via OpenRouter
- Free-tier: no per-token cost - automatically select best available free model
- 8B parameter instruct model — sufficient for factual Q&A with injected context
- Temperature = 0.1 for near-deterministic factual answers

### Prompt Strategy
```
SYSTEM: You are an HR policy assistant. ONLY use provided excerpts.
        Cite [Source: <title>] for every claim. Refuse out-of-scope questions.
        Max ~300 words.

USER: [Numbered excerpts 1–5 with title + source]
      ---
      Question: {question}
      Answer (with citations):
```
Key decisions:
- Numbered excerpts help the model produce accurate citation references
- Explicit refusal instruction handles out-of-scope guardrail
- Low temperature + short max_tokens limit hallucination and verbosity

### Web Application
Flask was chosen for its simplicity and suitability for the free-tier Render deployment. A single `gunicorn` worker is sufficient at the expected traffic level. CORS is enabled to allow future JS-only frontends.

### CI/CD
GitHub Actions installs dependencies, imports the app module (import smoke test), generates sample policies, and runs pytest on every push/PR. On pushes to `main` only, it fires the Render deploy webhook. This prevents broken code from reaching production.

---

## 2. Evaluation Approach & Results

### Evaluation Set
22 questions spanning all 10 policy topics plus 2 out-of-scope guardrail questions. Each question has a short gold answer (a key fact or phrase) for partial-match scoring.

### Metrics

| Metric | Definition | Measurement Method |
|---|---|---|
| **Groundedness** | % of answers factually consistent with retrieved context | LLM-as-judge (same model, 0-shot) |
| **Citation Accuracy** | % of answers whose citations point to supporting text | Keyword overlap heuristic: >5 shared tokens between answer and cited snippet |
| **Partial Match** | % of answers containing key phrases from gold answer | Token overlap ≥ 50% of gold tokens present in answer |
| **Latency p50/p95** | Request-to-answer wall-clock time | `time.perf_counter()` around retrieve + generate |

### Expected Results (benchmark targets)

| Metric | Target | Notes |
|---|---|---|
| Groundedness | ≥ 80% | Out-of-scope refusals counted as grounded |
| Citation Accuracy | ≥ 75% | Heuristic; can under-count when LLM paraphrases heavily |
| Partial Match | ≥ 70% | Short gold answers are strict; partial match is lenient |
| Latency p50 | < 2,000 ms | Dominated by OpenRouter free-tier LLM latency (~1–3 s) |
| Latency p95 | < 5,000 ms | Accounts for OpenRouter cold starts |

### Guardrail Evaluation
Questions outside the policy corpus (e.g., "What is the stock price?") should trigger the refusal phrase "I can only answer about our company policies." This is verified in the partial-match check for items tagged `GUARDRAIL`.

### Ablation Ideas (optional extensions)
| Dimension | Variants | Hypothesis |
|---|---|---|
| Top-k | k=3 vs k=5 vs k=8 | k=5 optimal; k=3 misses multi-clause answers, k=8 dilutes context |
| Chunk size | 256 vs 512 vs 1024 chars | 512 best; 256 loses clause context, 1024 wastes tokens on irrelevant text |
| Prompt | With / without explicit refusal instruction | Refusal instruction reduces false answers by ~15% |

### Limitations
- LLM-as-judge groundedness is evaluated by the same model that generated the answer — a stronger judge model would be more reliable
- Citation accuracy heuristic can miss citations when the LLM heavily paraphrases; a semantic similarity check would be more robust
- Free-tier OpenRouter has rate limits that can inflate p95 latency

---

