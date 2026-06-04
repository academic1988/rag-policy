# Design & Evaluation

## 1. Architecture & Design Decisions

### Document Corpus
Ten company policy documents were assembled covering PTO, remote work, expenses, code of conduct,
information security, holidays, performance reviews, leave of absence, data privacy, and learning
& development — totalling ~120 policy paragraphs. Documents are stored as `.txt` files and are
easily replaceable with or supplemented by PDFs.

### Chunking Strategy

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 512 characters | Balances context density with retrieval precision; policy clauses typically fit in 1–3 sentences |
| Overlap | 64 characters (~12%) | Prevents key facts from being stranded at chunk boundaries |
| Splitter | `RecursiveCharacterTextSplitter` | Respects paragraph and sentence boundaries before resorting to mid-sentence splits |

A character-based splitter was chosen over a token-based one to eliminate any external tokeniser
dependency at index time while remaining accurate enough for short policy sentences. The 12%
overlap ratio was chosen as a practical balance — large enough to preserve boundary context,
small enough to avoid redundant chunk storage.

### Embedding Model
**`all-MiniLM-L6-v2`** (SentenceTransformers, 22M parameters, 384-dimensional embeddings)
- Runs entirely locally — no API cost or rate limits during indexing or retrieval
- Delivers strong semantic similarity performance on English policy text
- Fast inference: ~1,000 sentences/second on CPU, making rebuild times negligible for this corpus size
- Module-level model caching (`_embedding_model` singleton in `retrieval.py`) ensures the model
  is loaded only once per process, avoiding repeated cold-start overhead on every query

### Vector Store
**ChromaDB (persistent local)**
- Zero infrastructure overhead — the collection is stored as a single directory on disk,
  making it compatible with free-tier hosting on Render without any external database
- Uses an HNSW index with cosine similarity, giving O(log n) approximate nearest-neighbour
  search — more than sufficient at this corpus size
- ChromaDB returns cosine *distance* (0 = identical, 1 = orthogonal); this is converted to
  *similarity* (`1.0 - distance`) in `retrieval.py` for intuitive scoring in the API response
- For larger corpora (>1M chunks) or multi-tenant production deployments, a managed vector
  store such as Pinecone or Weaviate would be more appropriate

### Retrieval
- **Top-k = 5** — retrieves enough context to cover multi-clause policy answers without
  exceeding a ~2,500-character context budget per query
- Chunks are sorted by cosine similarity score descending; a reranking hook exists in
  `retrieval.py` and can be swapped for a cross-encoder (e.g.,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) if latency budget and accuracy requirements increase

### LLM
**`openrouter/free`** via [OpenRouter](https://openrouter.ai)
- Automatically routes to the best available free-tier model, avoiding dependency on any
  single model that may be deprecated or rate-limited
- Configured with `temperature=0.1` for near-deterministic, factual answers
- `max_tokens=1024` caps response length to prevent runaway output on the free tier
- The model string is fully configurable via the `LLM_MODEL` environment variable, so
  switching to a specific model (e.g., `meta-llama/llama-3.3-70b-instruct:free`) requires
  no code changes

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
- Numbered, titled excerpts give the model unambiguous reference points for citation
- An explicit refusal instruction in the system prompt enforces the out-of-scope guardrail
  without relying on the model's default behaviour
- Low temperature and a token cap together reduce hallucination and verbosity
- Citation extraction in `generation.py` first attempts to match document titles mentioned
  in the answer, then falls back to the top-scored retrieved chunks if no explicit citation
  is found — ensuring a citation is always present in the response

### Web Application
Flask was chosen for its simplicity, minimal overhead, and well-documented compatibility
with free-tier Render deployments. The application exposes three endpoints:
- `GET /` — serves the chat UI (`templates/index.html`)
- `POST /chat` — the RAG pipeline endpoint (retrieve → rerank → generate → cite)
- `GET /health` — returns `{"status": "ok", "model": <model>}` for uptime monitoring

The ChromaDB collection is lazy-initialised on first use via `get_or_build_collection()` and
cached in a module-level variable for the lifetime of the process. A pre-warm attempt runs at
startup but is non-fatal, allowing the app to start cleanly in CI where no policy documents
are present. A single `gunicorn` worker is sufficient at expected free-tier traffic levels.
CORS is enabled to support future decoupled frontend deployments.

### CI/CD
The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push and pull request:
1. Installs dependencies from `requirements.txt`
2. Runs an import smoke test (`python -c "import app"`)
3. Generates sample policy documents
4. Runs the full `pytest` suite

Deployment to Render is triggered only on successful pushes to `main`, via a deploy webhook
stored as a GitHub repository secret (`RENDER_DEPLOY_HOOK_URL`). This ensures broken builds
never reach production.

---

## 2. Evaluation Approach & Results

### Evaluation Set
22 questions spanning all 10 policy topics, plus 2 out-of-scope guardrail questions. Each
question is paired with a short gold answer (a key fact or phrase) used for partial-match
scoring. The set is defined as a fixed list in `scripts/evaluate.py` and is run sequentially
to produce both per-question detail and aggregate summary metrics.

### Metrics

| Metric | Definition | Measurement Method |
|---|---|---|
| **Groundedness** | % of answers whose factual claims are fully supported by the retrieved context | LLM-as-judge: the same model is prompted to verdict YES/NO with a brief reason |
| **Citation Accuracy** | % of answers where at least one citation points to a chunk that substantively supports the answer | Keyword overlap heuristic: >5 shared tokens between the answer and the cited snippet |
| **Partial Match** | % of answers containing at least half the key tokens from the gold answer | Token presence check: `≥ 50%` of gold tokens found in the answer string |
| **Latency p50 / p95** | Request-to-answer wall-clock time across all 22 queries | `time.perf_counter()` wrapping the retrieve + generate steps |

### Benchmark Targets

| Metric | Target | Notes |
|---|---|---|
| Groundedness | ≥ 80% | Out-of-scope refusals are counted as grounded (no fabrication) |
| Citation Accuracy | ≥ 75% | Heuristic; may under-count when the LLM heavily paraphrases source text |
| Partial Match | ≥ 70% | Short gold answers make this a strict check; partial match threshold is intentionally lenient |
| Latency p50 | < 2,000 ms | Dominated by OpenRouter free-tier inference latency (~1–3 s typical) |
| Latency p95 | < 5,000 ms | Accounts for free-tier cold starts and occasional routing delays |

### Guardrail Evaluation
Out-of-scope questions (e.g., "What is the stock price of Acme Corp?") should trigger the
system prompt's refusal instruction. The partial-match check for `GUARDRAIL`-tagged items
looks for phrases such as "can only answer", "could not find", or "outside" in the response.
A successful refusal scores positively on both groundedness and partial match.

### Limitations
- **LLM-as-judge bias:** groundedness is judged by the same model that generated the answer.
  A stronger, independent judge model would produce more reliable verdicts.
- **Citation heuristic brittleness:** the keyword overlap check can miss valid citations when
  the LLM paraphrases heavily. A semantic similarity check between the answer and the cited
  snippet would be more robust.
- **Free-tier latency variance:** OpenRouter free-tier routing and cold starts can cause
  significant p95 inflation that is unrelated to the RAG pipeline itself.
- **Small corpus:** with only 10 documents, retrieval precision is high by default. Results
  may not generalise to larger, noisier corpora.

### Ablation Ideas (optional extensions)

| Dimension | Variants | Hypothesis |
|---|---|---|
| Top-k | k=3 vs k=5 vs k=8 | k=5 is optimal; k=3 misses multi-clause answers, k=8 dilutes the prompt with low-relevance chunks |
| Chunk size | 256 vs 512 vs 1024 chars | 512 is best; 256 loses clause context, 1024 wastes token budget on surrounding text |
| Prompt variant | With vs without explicit refusal instruction | The refusal instruction reduces out-of-scope false positives significantly |
| Reranking | Bi-encoder sort vs cross-encoder | Cross-encoder reranking expected to improve citation accuracy on ambiguous queries |

---

## 3. Technology Summary

| Component | Choice | Alternative to Consider |
|---|---|---|
| LLM | `openrouter/free` via OpenRouter | Specific pinned free models (e.g., Llama 3.1 8B, Mistral 7B) |
| Embeddings | `all-MiniLM-L6-v2` — local, SentenceTransformers | OpenAI `text-embedding-3-small` (requires API key + cost) |
| Vector DB | ChromaDB persistent local | Pinecone, Weaviate (require external infrastructure) |
| Orchestration | Manual (`requests` + LangChain text splitter only) | Full LangChain / LlamaIndex chain |
| Web framework | Flask | FastAPI (more overhead for this use case) |
| Hosting | Render free tier | Fly.io, Railway |
| CI/CD | GitHub Actions | GitLab CI, CircleCI |