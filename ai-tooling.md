# AI Tooling

## Tools Used

### Claude (Anthropic)
**Role:** Claude (claude.ai) and Claude Code (VS Code extension) were the primary AI tools used throughout this project, covering code generation, architecture planning, prompt engineering, and documentation.

**What worked well:**
- Generated complete, coherent module structures (ingestion, retrieval, generation) based on requirements, resulting in a well-structured codebase that required relatively minor corrections rather than a complete rewrite
- Produced accurate ChromaDB and SentenceTransformers integration code — correctly handled the cosine distance-to-similarity conversion and batch embedding
- Wrote the LLM-as-judge evaluation pattern correctly, including the correct JSON parsing logic for OpenRouter responses
- Drafted the GitHub Actions YAML with the correct conditional deploy step (`if: github.ref == 'refs/heads/main'`) without prompting
- Generated the system prompt with appropriate guardrail instructions that kept the model on-topic during testing

**What didn't work / needed iteration:**
- First version of `app.py` used a global `collection` variable that caused issues during pytest; needed to refactor to a lazy-init function
- The initial citation extraction logic used exact string matching which was too brittle — had to switch to case-insensitive substring matching
- Suggested `langchain` for orchestration but the manual approach (direct `requests` calls to OpenRouter) turned out simpler and more debuggable for this use case; dropped the LangChain dependency for the core pipeline
- The generated `app/` module folder conflicted with `app.py` (the Flask entry point) — Python resolved `app` to the file rather than the package, causing `ModuleNotFoundError` on import. Resolved by renaming the module folder to `rag/` and updating all imports accordingly
- The initially pinned dependency versions (`numpy==1.26.4`, `tiktoken==0.7.0`) were incompatible with Python 3.14, which lacked pre-built wheels for those versions and required a Rust compiler to build from source. Resolved by updating to `tiktoken==0.13.0` and loosening the numpy pin to `numpy>=1.26.4,<3.0.0` to allow pip to select a compatible pre-built wheel
- The `openrouter/auto` model string used in the initial configuration requires a paid OpenRouter account despite appearing to be a free routing option. Replaced with `openrouter/free`, which correctly routes to an available free-tier model automatically.

### Claude Code (VS Code extension)
**Role:** In-editor autocomplete while writing boilerplate (Flask routes, pytest fixtures, env loading).

**What worked well:**
- Excellent at completing repetitive patterns like dictionary unpacking, list comprehensions over chunk metadata
- Correctly suggested `response.raise_for_status()` after the `requests.post()` call

**What didn't work:**
- Suggestions for ChromaDB were outdated (suggested the deprecated `chromadb.Client()` instead of `chromadb.PersistentClient()`) — had to override manually

---

## Summary
AI tools accelerated development by approximately 60–70% on boilerplate and integration code. The main value was in generating correct API integration patterns (ChromaDB, SentenceTransformers, OpenRouter) and consistent docstrings/type hints. Human review was essential for catching outdated API suggestions, resolving Python version and dependency compatibility issues, fixing module naming conflicts, and tuning the RAG prompt for accurate refusals.