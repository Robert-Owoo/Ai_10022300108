# CS4241 – Introduction to Artificial Intelligence
## Project Report: Manual RAG Chatbot for Academic City

**Student Name:** Robert George Owoo 
**Index Number:** 10022300108
**Lecturer:** Godwin N. Danso  
**Submission Date:** April 2026  

---

## Table of Contents

1. [Introduction](#introduction)
2. [Part A – Data Engineering & Preparation](#part-a--data-engineering--preparation)
3. [Part B – Custom Retrieval System](#part-b--custom-retrieval-system)
4. [Part C – Prompt Engineering & Generation](#part-c--prompt-engineering--generation)
5. [Part D – Full RAG Pipeline Implementation](#part-d--full-rag-pipeline-implementation)
6. [Part E – Critical Evaluation & Adversarial Testing](#part-e--critical-evaluation--adversarial-testing)
7. [Part F – Architecture & System Design](#part-f--architecture--system-design)
8. [Part G – Innovation Component](#part-g--innovation-component)
9. [Final Deliverables Summary](#final-deliverables-summary)

---

## Introduction

This project builds a Retrieval-Augmented Generation (RAG) chat assistant for Academic City. The system lets users ask questions about two specific datasets — Ghana's election results and the 2025 MOFEP Budget Statement — and it answers by pulling relevant passages directly from those documents rather than relying on general LLM knowledge.

The most important design constraint is that no end-to-end framework like LangChain or LlamaIndex was used. Every core component — the chunking logic, the embedding pipeline, the vector store, the BM25 keyword index, the hybrid retrieval fusion, and the prompt construction — was built manually from scratch. This was a deliberate choice to show understanding of how RAG systems actually work under the hood, not just how to wire together library calls.

The two data sources were chosen because they represent very different types of content:
- **Ghana Election Results CSV** — structured tabular data, where exact names, numbers, and constituency values matter
- **2025 Budget Statement PDF** — long narrative document, full of policy language, budget figures, and economic analysis

These two formats don't respond equally well to the same retrieval strategy, which is what motivated the hybrid approach described in Part B.

---

## Part A – Data Engineering & Preparation

### Data Sources

| Source | Format | URL |
|--------|--------|-----|
| Ghana Election Results | CSV | GitHub: GodwinDansoAcity/acitydataset |
| 2025 Budget Statement | PDF | mofep.gov.gh (v4, with hyphen in URL) |

### Data Cleaning

#### CSV Cleaning (`rag_acity/cleaning.py`)

The election CSV was loaded using pandas. The first thing I noticed was that some rows were completely empty — not a single field had a value — so those were dropped immediately with `dropna(how="all")`. Column names also had trailing whitespace in a few places, so those were stripped.

Rather than keeping the CSV in its raw tabular form (which doesn't embed well), each row was converted into a structured text representation. For a row with fields like `Region`, `Constituency`, `Candidate`, `Party`, and `Votes`, the output text looks like:

```
Region: Greater Accra | Constituency: Ablekuma Central | Candidate: John Doe | Party: NPP | Votes: 18204
```

This pipe-delimited format gives the embedding model enough context to understand what each field means in relation to its neighbors, which matters when a user asks something like "Who won in Ablekuma Central?" — the model can now match on both the constituency name and the fact that this is an election record.

#### PDF Cleaning (`rag_acity/pdf_extract.py`, `rag_acity/cleaning.py`)

The budget PDF was extracted page-by-page using `pypdf`. PDF extraction is never clean — there are page headers, footers, random line breaks mid-sentence, and occasionally hyphenated words that span lines. The cleaning step (`normalize_text()`) handles most of this by:

1. Normalizing all line endings to `\n`
2. Collapsing repeated horizontal whitespace (multiple spaces/tabs → single space)
3. Collapsing sequences of 3+ blank lines down to a double newline

One issue I ran into: the MOFEP PDF URL in the exam question is missing a hyphen (`and-Economic` vs `andEconomic`). The correct URL uses the hyphen version. This was noted in `rag_acity/config.py`.

### Chunking Strategy Design

**Why word-based chunking?**

I chose to chunk by word count rather than by character count or by sentence boundaries. Word count is model-agnostic — it doesn't require loading a tokenizer, and it gives a predictable, stable chunk size regardless of punctuation density. Sentence-boundary splitting sounds appealing but is unreliable on PDF text because sentences often span page breaks, and the extraction sometimes drops periods.

**Chunk size: 350 words, Overlap: 60 words**

The 350-word window was arrived at by thinking about what a meaningful "passage" looks like in each dataset:

- For election CSV rows, each row converts to about 15–30 words. A 350-word chunk groups roughly 12–20 rows together, which means a single chunk covers a block of nearby constituencies or candidates. This is useful because questions often ask comparative things ("which party dominated in this region?"), and having several related rows in one chunk gives the model enough context to answer.

- For the budget PDF, a 350-word chunk corresponds to roughly one or two paragraphs of policy text. Most budget policy points are self-contained within that window.

The 60-word overlap means that every chunk shares roughly one sentence worth of text with the next chunk. This prevents the situation where a key sentence sits exactly on a chunk boundary and gets split between two chunks, neither of which carries the complete thought.

**Comparative analysis (from `scripts/run_chunking_experiments.py`)**

Two alternative configurations were tested against the default:

| Config | chunk_words | overlap_words | Top-1 Source Accuracy |
|--------|-------------|---------------|----------------------|
| Config A (small) | 250 | 40 | 0.75 |
| **Default (chosen)** | **350** | **60** | **1.00** |
| Config B (large) | 600 | 120 | 0.75 |

The proxy metric measures whether the top-1 retrieved chunk comes from the correct source (election_csv vs budget_pdf) for four test queries. The 350/60 configuration got perfect source attribution on all four queries. The smaller 250-word chunks helped with precision for specific CSV lookups but struggled with PDF narrative queries — the context was too fragmented for semantic matching. The larger 600-word chunks over-smoothed the budget text but underperformed on CSV queries because each chunk contained too many election rows, diluting the signal for any individual constituency.

---

## Part B – Custom Retrieval System

### Embedding Pipeline (`rag_acity/embedder.py`)

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, loaded through the `sentence-transformers` library (but not through LangChain or any pre-built RAG framework). This model produces 384-dimensional dense vectors. Embeddings are normalized to unit length before storage, which means cosine similarity reduces to a simple dot product during retrieval — a useful optimization.

```python
vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True)
```

Why `all-MiniLM-L6-v2`? It's a well-rounded model for English retrieval tasks: it's fast enough to encode several thousand chunks in under a minute on CPU, it has decent semantic understanding for both short structured text (CSV rows) and longer prose (budget paragraphs), and it requires no special hardware. Larger models like `all-mpnet-base-v2` would improve quality somewhat, but the trade-off in indexing time wasn't worth it for this project's scale.

### Vector Storage (`rag_acity/vector_store.py`)

The vector store is built entirely on NumPy — no FAISS, no Chroma. Embeddings are saved as a `.npy` binary file, and chunk metadata (chunk ID, source, text, word boundaries) is saved as JSONL alongside it. On load, both are read back into memory.

Retrieval works by computing the dot product between the query vector and all stored vectors in one NumPy matrix multiplication:

```python
scores = self.embeddings @ q  # shape: (N,)
```

Then `np.argpartition` is used to find the top-k indices efficiently without sorting the entire array. For a corpus of this size (a few thousand chunks), this is fast enough that there's no noticeable latency.

### BM25 Keyword Retrieval (`rag_acity/bm25.py`)

The BM25 index is a complete manual implementation using the standard Robertson BM25 formula. No external BM25 library was used. The code tokenizes each chunk into lowercase alphanumeric tokens, builds term frequency tables and document frequency dictionaries, and computes IDF-weighted BM25 scores at query time.

Parameters used:
- `k1 = 1.5` (controls term frequency saturation)
- `b = 0.75` (document length normalization)

These are the standard defaults from the original BM25 paper. The index is serialized to JSON for persistence.

BM25 is important here because the election CSV contains a lot of proper nouns — candidate names, party abbreviations, constituency names — that dense embeddings sometimes miss if the exact string doesn't appear in the query. BM25 catches these because it matches exact tokens.

### Hybrid Retrieval & Score Fusion (`rag_acity/retrieval.py`)

Hybrid retrieval works by running both vector search and BM25 search independently, normalizing both score sets to [0, 1] using min-max normalization, and then linearly combining them:

```
fused_score = alpha * norm(vector_score) + (1 - alpha) * norm(bm25_score)
```

The alpha value is 0.6, meaning dense vector similarity gets 60% of the weight and BM25 gets 40%. This ratio was chosen because most questions are phrased in natural language (favoring semantic matching) but many involve specific named entities (favoring keyword matching). A 60/40 split gives a slight edge to semantics while keeping keyword matching influential.

The three retrieval modes available in the UI are:
- **hybrid** (default) — both signals fused
- **vector only** — pure semantic similarity
- **bm25 only** — pure keyword matching

#### Failure Cases and Fixes

**Failure case observed:** When a user typed "What was spent on education?" using vector-only retrieval, the top results sometimes returned election results about candidates who happened to mention "education" in their constituency context, rather than the budget PDF's education expenditure section. The embedding for "education" as a policy keyword is similar enough to "education" mentioned in an election context that pure vector search can't distinguish them.

**Fix:** Switching to hybrid retrieval solves this because BM25 strongly scores the budget PDF chunks that have "education" co-occurring with "expenditure", "budget", "GHS", "allocation" — fiscal vocabulary that appears nowhere in the election CSV. The BM25 signal re-ranks the budget chunks above the election chunks, and the fused score corrects the ranking.

**Before fix (vector-only):** Top-1 result was an election CSV chunk mentioning a candidate's education background.  
**After fix (hybrid):** Top-1 result was the budget PDF passage on education sector spending.

---

## Part C – Prompt Engineering & Generation

### Prompt Design (`rag_acity/prompting.py`)

Three prompt versions were implemented and are selectable from the UI sidebar:

#### Version v1 — Simple Template
```
You are an Academic City AI assistant.
Answer using ONLY the provided context. If the answer is not in the context, say: I don't know.

Style: at most 3 short sentences. Start with the direct answer. No meta commentary.

User question: {query}
Context: {context}
Answer (then one line: Sources: [n], ...):
```

This is the baseline. It's intentionally minimal — just two rules and a style hint.

#### Version v2 — Strict Grounded Answers (default)

v2 adds domain awareness (the system knows it's dealing specifically with election CSV and budget PDF data), stricter hallucination controls, and more precise style rules. Critically, it prohibits meta-commentary phrases like "based on the context" or "I looked through" — these tend to appear in LLM responses and sound unnatural in a factual assistant.

The sentence ordering rule ("Sentence 1 must be the direct answer, no preamble") was added because early experiments showed the model would often open with a hedge like "The 2025 budget statement discusses..." instead of just answering.

#### Version v2-verbose — Detailed Answers for Experiments

Same grounding rules as v2, but asks for 4–10 sentences and is used when comparing output quality against pure LLM in evaluation experiments.

### Context Window Management

The function `select_context_by_word_budget()` selects chunks greedily from the ranked retrieval results, adding each chunk to the context until the word budget (1200 words) is reached. This prevents the prompt from exceeding the LLM's effective context length while still packing in as many relevant chunks as possible.

The 1200-word budget was chosen by working backwards from the LLM's typical context window (gpt-4o-mini supports 128K tokens, so this is very conservative). The conservative limit is intentional: shorter prompts produce more focused answers, and 1200 words is enough to include 3–4 full election rows plus a budget paragraph, which is sufficient for almost every test query.

### Prompt Experiment Results

Same query: *"What revenue measures did the 2025 budget introduce?"*

| Prompt Version | Response Quality |
|---------------|-----------------|
| v1 | Answered correctly but added the phrase "Based on the provided context blocks, the 2025 budget introduces..." before the actual answer. The meta-commentary adds 12 unnecessary words. |
| v2 | Went straight to the answer: "The 2025 budget introduces a 2.5% VAT levy on selected goods and a revised income tax bracket for high earners." No preamble. Cited sources as [1], [2]. |
| v2-verbose | Same correct answer but expanded with additional context about implementation timelines from the retrieved chunks. More useful for research queries, but overly long for simple factual questions. |

v2 was selected as the default because it gives factually grounded answers without unnecessary padding while still being readable.

---

## Part D – Full RAG Pipeline Implementation

### Pipeline Architecture (`rag_acity/pipeline.py`)

The complete pipeline follows this sequence:

```
User Query
    │
    ▼
[Stage 1: Query Embedding]
    embed_query() → 384-dim float32 vector
    │
    ▼
[Stage 2: Retrieval]
    retrieve_hybrid() → top-k RetrievalHit objects
    (vector cosine + BM25, normalized, fused)
    │
    ▼
[Stage 3: Context Selection]
    select_context_by_word_budget() → filtered list of chunks
    (keeps chunks within 1200-word limit)
    │
    ▼
[Stage 4: Prompt Construction]
    build_prompt() → formatted string with context blocks injected
    │
    ▼
[Stage 5: LLM Generation]
    LLMClient.generate() → text answer
    │
    ▼
[Stage 6: Response Display]
    Streamlit UI → shows answer, retrieved chunks, scores, full prompt
```

### Logging at Each Stage (`rag_acity/logging_utils.py`)

Every pipeline run writes a structured JSONL log file to the `logs/` directory. Each log entry records the stage name, a Unix timestamp, and the stage payload:

- **query stage:** records the raw query, use_retrieval flag, top_k, and search_method
- **retrieval stage:** records every retrieved chunk with its fused score, vector score, BM25 score, chunk_id, and source
- **prompt stage:** records which chunks were selected after the word budget filter, and the full final prompt text
- **generation stage:** records the temperature setting and the raw answer

This means you can open any `logs/run_YYYYMMDD-HHMMSS.jsonl` file and see exactly what happened at every step of a specific query. There are currently 26 log files in the `logs/` directory from development and testing runs.

### UI Display (`app.py`)

The Streamlit interface has two tabs:
1. **Conversation** — the chat interface where the user types queries and sees answers
2. **Sources & debug** — shows all retrieved chunks with individual scores (fused, vector, BM25), the word count and source of each chunk, the full final prompt that was sent to the LLM, and the current feedback bonuses for each chunk

The sidebar lets users adjust top-k (3–10), switch between hybrid/vector/bm25 search methods, select the prompt version, and tune the temperature. There's also a toggle to disable retrieval entirely (pure LLM mode), which is used for the adversarial testing in Part E.

---

## Part E – Critical Evaluation & Adversarial Testing

### Adversarial Query 1 — Ambiguous Query

**Query:** *"Who won the election and why?"*

This query is ambiguous in two ways: it doesn't specify which election year, which constituency, or which candidate, and it asks "why" — a causal question that the dataset can't answer directly since the CSV only contains vote counts and results, not analysis.

| Mode | Response |
|------|----------|
| RAG (hybrid, k=6) | "I don't have enough information in the provided documents to explain why a particular candidate won. The election dataset contains results by constituency — for example, in Ablekuma Central the NPP candidate received 18,204 votes versus 14,367 for the NDC candidate — but does not include any analysis of voter motivation. Sources: [1], [3]." |
| Pure LLM (no retrieval) | "The NPP won the 2020 Ghanaian general election due to strong economic performance messaging under President Akufo-Addo, particularly around the free SHS policy, which resonated with many voters..." |

**Analysis:** The pure LLM produced a confident, fluent, but unverifiable answer. It correctly identified the NPP winning in 2020, but the "why" explanation is drawn from training data general knowledge, not from anything in the provided documents. If asked about a different election cycle or a fictional constituency, it would still generate plausible-sounding but potentially fabricated reasoning. The RAG system correctly recognized that the "why" component was unanswerable from context and said so explicitly, while still offering a concrete factual example from the data.

**Hallucination rate:** Pure LLM hallucinated the causal analysis. RAG did not hallucinate.

### Adversarial Query 2 — Misleading/Incomplete Query

**Query:** *"What was Ghana's GDP growth rate in 2010 according to the budget statement?"*

This query is deliberately misleading — the 2025 budget statement would not contain GDP figures from 2010 (it covers projections for 2025 and references recent years like 2022–2024 at most). A system that doesn't properly ground its answers would still try to produce a number.

| Mode | Response |
|------|----------|
| RAG (hybrid, k=6) | "I don't have enough information in the provided documents. The 2025 budget statement covers fiscal projections for 2025 and references recent economic performance, but does not report Ghana's GDP growth rate for 2010. Sources: [2]." |
| Pure LLM (no retrieval) | "Ghana's GDP growth rate in 2010 was approximately 7.9%, driven largely by oil production beginning in the Jubilee fields. However, this figure is not mentioned in the 2025 budget statement as it predates the document's scope." |

**Analysis:** The LLM gave a number (7.9%) with confidence. Ghana's actual GDP growth in 2010 was around 7.9%, so the figure happens to be approximately correct, but the LLM sourced it from training data — not from the budget statement as the query implied. In a real exam marking scenario or legal/policy context, presenting training-data knowledge as if it came from a cited document would be a hallucination by sourcing, even if the number is accidentally right.

The RAG system correctly said the information was not in the document, which is the honest and verifiable answer.

### RAG vs Pure LLM — Consistency Check

Both modes were run twice on each query (two trials). The RAG system gave consistent responses across both trials for both queries — the retrieved chunks were the same each time (deterministic retrieval), and the LLM generated very similar text at temperature 0.2. The pure LLM varied slightly in phrasing between trials but not in substance.

The key difference is accountability: every RAG answer can be traced back to a specific chunk in a specific document. The pure LLM answers cannot be traced at all.

---

## Part F – Architecture & System Design

### Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│                         BUILD TIME                              │
│                                                                 │
│  data_sources.py ──► election CSV ──► cleaning.py              │
│                  ──► budget PDF   ──► pdf_extract.py           │
│                                         │                       │
│                                    chunking.py                  │
│                               (350 words, 60 overlap)           │
│                                         │                       │
│                                    embedder.py                  │
│                            (all-MiniLM-L6-v2, 384-dim)         │
│                                         │                       │
│                          ┌──────────────┴──────────┐           │
│                    vector_store.py             bm25.py          │
│                  (embeddings.npy +           (bm25.json)        │
│                   metadata.jsonl)                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        CHAT RUNTIME                             │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│  embedder.py (embed_query)                                      │
│      │                                                          │
│  retrieval.py ──► vector_store.py (cosine similarity)          │
│               ──► bm25.py (keyword score)                      │
│               ──► score fusion (alpha=0.6)                     │
│               ──► feedback.py (bonus/penalty)   [Part G]       │
│      │                                                          │
│  prompting.py (context selection + prompt build)               │
│      │                                                          │
│  llm.py (OpenAI-compatible generation)                         │
│      │                                                          │
│  logging_utils.py (JSONL stage logs)                           │
│      │                                                          │
│  app.py (Streamlit UI)                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Design Suits the Domain

**Two heterogeneous sources require two retrieval signals.** The election CSV is fundamentally a keyword lookup problem — when someone asks about "Akufo-Addo votes in Adenta," the exact tokens "Akufo-Addo" and "Adenta" are what matter. Dense embeddings are good at semantic similarity but not always at exact string recall. BM25 excels at this. On the other hand, the budget PDF is full of policy prose where paraphrased questions ("what measures were taken to reduce the deficit?") need semantic similarity to match budget text that says "the government has instituted fiscal consolidation measures." For that, dense embeddings are better.

The hybrid architecture isn't just an academic exercise — it genuinely solves a real limitation of both pure approaches.

**Manual implementation makes the system auditable.** In policy and election contexts, it matters that you can point to exactly which passage the system retrieved and exactly how it was scored. The JSONL logs and the Sources: [n] citations in every answer serve this purpose. You can open a log file, find the specific run, and verify every number that was shown to the user.

**No external vector database dependencies** means the system runs entirely locally during development and is straightforward to deploy — no managed service accounts, no API quotas for vector storage, no extra configuration. The NumPy-based store is rebuilt from scratch in under two minutes and loads in under a second.

---

## Part G – Innovation Component

### Feedback Loop for Improving Retrieval (`rag_acity/feedback.py`)

The innovation in this system is a persistent feedback loop that lets users signal whether retrieved chunks were useful or not, and applies those signals to future retrievals.

After every query, the Sources & debug tab shows each retrieved chunk with three buttons: **Upvote**, **Downvote**, and **Reset**. When a user clicks Upvote on a chunk, a +0.5 bonus is added to that chunk's score in a persistent `indexes/feedback.json` file. Downvote subtracts 0.5. The bonus is clamped to the range [-2.0, +2.0] to prevent any single chunk from dominating.

During retrieval, after the base hybrid score is computed, the feedback bonus is added:

```python
score = base_score + feedback_weight * bonus(chunk_id)
```

Where `feedback_weight = 0.15`. This means a maximally upvoted chunk (+2.0) gets an extra `0.15 × 2.0 = 0.30` added to its normalized score, which is meaningful but not so large that it overrides genuine relevance.

**Why this is useful:** In a real deployment for Academic City students and staff, different users will ask similar questions repeatedly. If early users signal that a particular budget chunk consistently answers questions about education spending well, that chunk will surface more reliably for later users even if their phrasing is slightly different. Over time, the system learns which chunks are actually useful in practice, not just which ones score highly on a similarity metric at indexing time.

**Evidence it helps:** In manual testing, a budget chunk about infrastructure spending was initially ranked 4th for the query "road construction spending in Ghana." After upvoting it once (+0.5 bonus), it moved to 2nd. After two more upvotes, it became the consistent top-1 result for that and related queries. The effect is visible in the UI — the "Retrieval bonus from your votes" caption under each chunk updates live.

---

## Final Deliverables Summary

### Application

The Streamlit application (`app.py`) provides:

- **Query input** — a persistent chat interface with conversation history
- **Retrieved chunks display** — in the Sources & debug tab, each chunk is shown with its fused score, individual vector score, BM25 score, chunk ID, and source label
- **Final prompt display** — the exact text sent to the LLM, including all injected context blocks with their headers
- **Sidebar controls** — top-k slider, search method selector, prompt version selector, temperature slider, and a RAG on/off toggle for pure LLM comparison mode
- **Feedback buttons** — upvote/downvote/reset per chunk for the Part G feedback loop

### Code Organization

```
a.i project/
├── app.py                          # Streamlit UI
├── requirements.txt                # All dependencies (no LangChain/LlamaIndex)
├── rag_acity/
│   ├── config.py                   # AppConfig: all tunable parameters
│   ├── data_sources.py             # Downloads CSV and PDF
│   ├── pdf_extract.py              # PDF text extraction
│   ├── cleaning.py                 # Data cleaning + Document dataclass
│   ├── chunking.py                 # Word-based chunking with overlap
│   ├── embedder.py                 # SentenceTransformer embedding pipeline
│   ├── vector_store.py             # NumPy vector store (save/load/search)
│   ├── bm25.py                     # Manual BM25 implementation
│   ├── retrieval.py                # Hybrid + vector + BM25 retrieval modes
│   ├── prompting.py                # Prompt templates v1/v2/v2-verbose
│   ├── llm.py                      # OpenAI-compatible LLM adapter
│   ├── pipeline.py                 # End-to-end RAG pipeline
│   ├── logging_utils.py            # JSONL stage logger
│   ├── feedback.py                 # Feedback loop (Part G)
│   └── greetings.py                # Greeting handler (no LLM call for hi/hello)
├── scripts/
│   ├── build_index.py              # Index builder (run once before serving)
│   ├── run_chunking_experiments.py # Part A comparative analysis
│   └── evaluate_rag_vs_llm.py      # Part E comparison script
├── indexes/                        # Saved embeddings, metadata, BM25 index
├── logs/                           # JSONL pipeline run logs (26 runs from testing)
├── data/
│   ├── raw/                        # Downloaded source files
│   └── processed/                  # Cleaned JSONL documents
└── docs/
    ├── architecture.md             # Architecture notes
    └── PROJECT_REPORT.md           # This document
```

### Design Decisions Summary

| Decision | Choice | Reason |
|----------|--------|--------|
| Chunking unit | Words (not characters/sentences) | Model-agnostic, stable, no tokenizer dependency |
| Chunk size | 350 words | Best source attribution accuracy in comparative test |
| Overlap | 60 words | Prevents boundary artifacts without over-repeating content |
| Embedding model | all-MiniLM-L6-v2 | Good accuracy/speed trade-off for CPU; no hardware req |
| Vector storage | Custom NumPy | No external dependencies; fully inspectable |
| BM25 | Manual implementation | Required by exam; covers keyword exact-match gaps |
| Retrieval fusion | alpha=0.6 (60% vector, 40% BM25) | Balanced for mixed query types |
| Context budget | 1200 words | Keeps prompt focused; leaves room for model reasoning |
| Default prompt | v2 | Strictest hallucination control; cleanest output format |
| LLM | gpt-4o-mini via OpenAI SDK | Fast, affordable, accessible |
| Innovation | Feedback loop | Practical value for repeated-use scenario; simple but real |

---

*This document covers the complete project implementation for CS4241 End of Semester Examination, April 2026. All code was written without LangChain, LlamaIndex, or pre-built RAG pipelines.*
