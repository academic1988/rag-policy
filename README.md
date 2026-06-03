# Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions about company policies using a ChromaDB vector store, sentence-transformer embeddings, and an LLM served via [OpenRouter](https://openrouter.ai).

---

## Quick Start

### Prerequisites
- Python 3.10 or 3.11
- An [OpenRouter](https://openrouter.ai) API key (free tier works)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/policy-rag.git
cd policy-rag
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=<your key>
```

### 5. Generate sample policy documents
```bash
python scripts/generate_sample_policies.py
```
This creates 10 `.txt` policy files in `data/policies/`.
You can replace or supplement them with your own PDFs/txt/md files.

### 6. Build the vector index
```bash
python scripts/build_index.py
```
Use `--force` to rebuild from scratch.

### 7. Run the application
```bash
python app.py
```
Open http://localhost:5000 in your browser.

---

## API Reference

### `GET /`
Web chat interface.

### `GET /health`
```json
{ "status": "ok", "model": "meta-llama/llama-3.1-8b-instruct:free" }
```

### `POST /chat`
**Request:**
```json
{ "question": "How many PTO days do I get?" }
```
**Response:**
```json
{
  "answer": "Full-time employees receive 15 days of PTO per year... [Source: PTO Policy]",
  "citations": [
    {
      "title": "Pto Policy",
      "source": "pto-policy.txt",
      "snippet": "Years 0–2: 15 days per year…",
      "score": 0.923
    }
  ],
  "latency_ms": 1420,
  "model": "meta-llama/llama-3.1-8b-instruct:free",
  "chunks_retrieved": 5
}
```

---

## Running Tests
```bash
pytest tests/ -v
```

---

## Evaluation
```bash
python scripts/evaluate.py
```
Generates `data/eval_report.json` with groundedness, citation accuracy, partial match, and latency metrics across 22 questions.

---

## Adding Your Own Policy Documents
Place `.pdf`, `.txt`, or `.md` files in `data/policies/`, then rebuild the index:
```bash
python scripts/build_index.py --force
```

---

## Deployment (Render)

1. Push this repo to GitHub.
2. Create a new **Web Service** on [Render](https://render.com) pointing to the repo.
3. Render will auto-detect `render.yaml` and configure the service.
4. Set `OPENROUTER_API_KEY` in Render's environment variable settings.
5. Add your Render deploy hook URL as a GitHub secret `RENDER_DEPLOY_HOOK_URL`.

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) will:
- Install dependencies
- Run a build check and tests on every push/PR
- Trigger a Render deployment on pushes to `main`

---

## Project Structure
```
policy-rag/
├── app/
│   ├── ingestion.py      # PDF/text loading, chunking, embedding, ChromaDB indexing
│   ├── retrieval.py      # Query embedding + vector search
│   └── generation.py     # Prompt building + OpenRouter LLM call
├── data/
│   └── policies/         # Policy documents (PDF/txt/md)
├── scripts/
│   ├── generate_sample_policies.py
│   ├── build_index.py
│   └── evaluate.py
├── templates/
│   └── index.html        # Chat UI
├── tests/
│   └── test_app.py
├── .github/workflows/
│   └── ci-cd.yml
├── app.py                # Flask application entry point
├── requirements.txt
├── render.yaml
├── Procfile
└── .env.example
```
