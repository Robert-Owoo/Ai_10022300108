# Academic City RAG Chatbot (Manual RAG Implementation)

**Student Name:** Robert George Owoo  
**Index Number:** 10022300108


## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** chat assistant for Academic City using:

- **Ghana Election Results CSV** (provided dataset)
- **MOFEP 2025 Budget Statement PDF** (provided dataset)

**Important constraint satisfied:** No end-to-end RAG frameworks (no LangChain, LlamaIndex, or pre-built RAG pipelines). Core RAG components are implemented manually:

- Data cleaning
- Chunking (with configurable size/overlap)
- Embedding pipeline (Sentence Transformers)
- Vector similarity + top-k retrieval
- **Hybrid retrieval** (keyword BM25 + vector)
- Prompt construction + context window management
- Full pipeline logging + UI display of retrieved chunks/scores/final prompt

## Architecture

```mermaid
flowchart LR
  A[User Query] --> B[Query Prep]
  B --> C[Hybrid Retrieval<br/>(BM25 + Vector)]
  C --> D[Top-k Chunks + Scores]
  D --> E[Context Selection<br/>(truncate / filter)]
  E --> F[Prompt Template<br/>(hallucination controls)]
  F --> G[LLM]
  G --> H[Answer + Citations]
  C --> L[Stage Logs]
  E --> L
  F --> L
  G --> L
```

## What’s inside
- **Streamlit UI**: `app.py`
- **Index building**: `scripts/build_index.py`
- **Chunking experiments**: `scripts/run_chunking_experiments.py`
- **RAG vs pure-LLM evaluation**: `scripts/evaluate_rag_vs_llm.py`
- **Manual experiment log template**: `experiments/manual_log.md`
- **Architecture explanation**: `docs/architecture.md`

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure API key (for generation)
This app uses an OpenAI-compatible API for generation.

- Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-4o-mini"
```

If you don’t have an API key, you can still run indexing + retrieval, but generation will fail.

## Build the index (downloads datasets, cleans, chunks, embeds)

Run this **from the project root** so Python can import `rag_acity`:

```powershell
cd "C:\Users\kappi\OneDrive\Desktop\a.i project"
.\.venv\Scripts\python -m scripts.build_index
```

(Running `python scripts/build_index.py` directly often fails with `ModuleNotFoundError: rag_acity` because the working directory on `sys.path` is not what we need.)

Outputs:
- `data/processed/` cleaned documents
- `indexes/` embeddings + metadata

## Run the app (web GUI in your browser)

**Option A — PowerShell launcher (Windows):**

```powershell
cd "C:\Users\kappi\OneDrive\Desktop\a.i project"
powershell -ExecutionPolicy Bypass -File .\run_web.ps1
```

**Option B — manual:**

```powershell
cd "C:\Users\kappi\OneDrive\Desktop\a.i project"
.\.venv\Scripts\python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). That is your **web app** for chatting with the assistant.

In the UI you can:
- Ask questions
- See retrieved chunks + similarity scores
- See the final prompt sent to the LLM
- Compare **RAG vs pure LLM**

## Experiments (required deliverables)

### Chunking impact

```bash
python scripts/run_chunking_experiments.py
```

### RAG vs pure LLM

```bash
python scripts/evaluate_rag_vs_llm.py
```

### Manual experiment logs (must be manual)
Fill in:
- `experiments/manual_log.md`

## Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub as `ai_<your_index_number>`
2. Go to Streamlit Community Cloud and deploy the app with:
   - **Main file path**: `app.py`
3. Add secrets (in Streamlit settings):
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL` (optional)

## Submission checklist (per exam instructions)
- GitHub repo: `ai_<index_number>`
- Deployed URL (Streamlit Cloud)
- Video walkthrough (≤ 2 minutes)
- Manual experiment logs
- Detailed documentation
- Invite collaborator: `godwin.danso@acity.edu.gh` or `GodwinDansoAcity`

