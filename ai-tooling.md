# AI Tooling

## Tools Used

### Claude (Anthropic)
**Role:** Primary code generation, architecture planning, prompt engineering, documentation.

**What worked well:**
- Generated complete, coherent module structures (ingestion, retrieval, generation) in a single pass that required minimal editing
- Produced accurate ChromaDB and SentenceTransformers integration code — correctly handled the cosine distance-to-similarity conversion and batch embedding
- Wrote the LLM-as-judge evaluation pattern correctly on first attempt, including the correct JSON parsing logic for OpenRouter responses
- Drafted the GitHub Actions YAML with the correct conditional deploy step (`if: github.ref == 'refs/heads/main'`) without prompting
- Generated the system prompt with appropriate guardrail instructions that kept the model on-topic during testing

**What didn't work / needed iteration:**
- First version of `app.py` used a global `collection` variable that caused issues during pytest; needed to refactor to a lazy-init function
- The initial citation extraction logic used exact string matching which was too brittle — had to switch to case-insensitive substring matching
- Suggested `langchain` for orchestration but the manual approach (direct `requests` calls to OpenRouter) turned out simpler and more debuggable for this use case; dropped the LangChain dependency for the core pipeline

### GitHub Copilot
**Role:** In-editor autocomplete while writing boilerplate (Flask routes, pytest fixtures, env loading).

**What worked well:**
- Excellent at completing repetitive patterns like dictionary unpacking, list comprehensions over chunk metadata
- Correctly suggested `response.raise_for_status()` after the `requests.post()` call

**What didn't work:**
- Copilot suggestions for ChromaDB were outdated (suggested the deprecated `chromadb.Client()` instead of `chromadb.PersistentClient()`) — had to override manually

---

## Summary
AI tools accelerated development by approximately 60-70% on boilerplate and integration code. The main value was in generating correct API integration patterns (ChromaDB, SentenceTransformers, OpenRouter) and consistent docstrings/type hints. Human review was essential for: catching outdated API suggestions, refactoring for testability, and tuning the RAG prompt for accurate refusals.
