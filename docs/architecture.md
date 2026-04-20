Student Name: Robert George Owoo  
Index Number: 10022300108  

## Architecture & System Design (Part F)

### Components
- **Data Sources**
  - `Ghana_Election_Result.csv` (tabular election records)
  - `2025-Budget-Statement-and-Economic-Policy_v4.pdf` (long narrative PDF; MOFEP URL uses hyphens)

- **Ingestion + Cleaning**
  - `rag_acity/data_sources.py`: downloads datasets
  - `rag_acity/pdf_extract.py`: extracts PDF text (page-by-page)
  - `rag_acity/cleaning.py`: normalizes whitespace; converts CSV rows into canonical text records

- **Chunking**
  - `rag_acity/chunking.py`: word-based chunking with overlap
  - Rationale: chunking reduces the retrieval unit size; overlap preserves continuity for long PDF passages.

- **Embedding**
  - `rag_acity/embedder.py`: SentenceTransformers encoder producing normalized dense vectors

- **Retrieval**
  - Vector retrieval: `rag_acity/vector_store.py` (cosine similarity via dot product on normalized vectors)
  - Keyword retrieval: `rag_acity/bm25.py` (manual BM25 implementation)
  - Hybrid retrieval: `rag_acity/retrieval.py` (score fusion)

- **Prompt + Generation**
  - `rag_acity/prompting.py`: context selection (word budget) + prompt templates (v1/v2)
  - `rag_acity/llm.py`: OpenAI-compatible LLM adapter

- **Pipeline + Logging**
  - `rag_acity/pipeline.py`: end-to-end RAG pipeline
  - `rag_acity/logging_utils.py`: structured JSONL stage logs

- **UI**
  - `app.py`: Streamlit app showing retrieved chunks, scores, and final prompt

### Data flow
1. **Build index** (`scripts/build_index.py`)
   - download → extract → clean → chunk → embed → save embeddings + metadata → build BM25
2. **Chat runtime** (`app.py`)
   - user query → embed query → hybrid retrieval → context selection → prompt build → LLM → response
3. **Observability**
   - Every run writes stage logs: query, retrieval hits, final prompt, and generation output.

### Why this design fits the domain
- **Two heterogeneous sources** (tabular CSV + long PDF) benefit from hybrid retrieval:
  - CSV facts often require **keyword/exact matching** (names, constituencies, numeric fields)
  - PDF narrative benefits from **semantic similarity**
- **Manual implementation** makes the RAG mechanics inspectable, debuggable, and align with exam constraints.

