# Manual Experiment Log

**Student Name:** Robert George Owoo  
**Index Number:** 10022300108  

> This log records observations made while running and testing the RAG system between 16–17 April 2026.
> All scores, answers, and chunk data are copied directly from the JSONL logs in `logs/`.
> Interpretations are my own.

---

## Environment

- **Date/time:** 16 April 2026, starting ~18:42 (WAT)
- **Machine/OS:** Windows 11 Pro (23H2)
- **Python version:** 3.13 (venv)
- **Embedding model:** sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **LLM:** llama-3.1-8b-instant via Groq API (OpenAI-compatible endpoint)
- **Chunking config:** chunk_words=350, overlap_words=60 (default)

---

## Part A: Cleaning + Chunking

### Cleaning notes (CSV)

I opened `Ghana_Election_Result.csv` after downloading and found the following columns:
`Year`, `Old Region`, `New Region`, `Code`, `Candidate`, `Party`, `Votes`, `Votes(%)`.

A few rows were completely empty (no values in any column) — these were dropped with `dropna(how="all")`.
Column names had occasional trailing whitespace which was stripped. No numeric columns had obviously wrong types.

Each row was converted into pipe-delimited text, e.g.:  
`Year: 2012 | Old Region: Western Region | New Region: Western North Region | Code: NDC | Candidate: John Dramani Mahama | Party: NDC | Votes: 221651 | Votes(%): 62.95%`

This format was chosen so the embedding model has enough field context — just a bare number like `221651` doesn't embed meaningfully on its own, but `Votes: 221651` does.

### Cleaning notes (PDF)

The 2025 Budget Statement PDF extracted mostly cleanly using `pypdf`. The main issues I noticed:
- Page headers and footers sometimes appeared as standalone lines between paragraphs
- Occasional mid-sentence line breaks where the PDF rendering split a long line
- Some section numbers appeared as isolated lines (e.g., "3.1" floating without its heading text)

The `normalize_text()` function collapses these by stripping repeated whitespace and merging excess blank lines. I verified the cleaned text by checking a few budget section chunks manually after indexing.

One concrete issue: the exam PDF URL for the budget statement uses `andEconomic` without a hyphen, but the actual MOFEP server URL requires `and-Economic` (with hyphen). I caught this when the download returned a 404, then corrected the URL in `rag_acity/config.py`.

### Chunking strategy

- **Chunk size:** 350 words  
- **Overlap:** 60 words  
- **Justification:** Each election CSV row converts to roughly 20–25 words, so 350 words groups about 14–17 rows — enough for regional comparisons without mixing too many regions. For the PDF, 350 words covers 1–2 policy paragraphs, which keeps policy topics self-contained. The 60-word overlap (~1 sentence) ensures a key sentence at a boundary appears in both adjacent chunks.

### Chunking comparative run

Ran: `python -m scripts.run_chunking_experiments` (16 April 2026)

Test queries and expected source used:
1. "Which party won in a specific constituency?" → expected: election_csv
2. "What are key fiscal measures mentioned in the 2025 budget?" → expected: budget_pdf
3. "Mention some revenue policy proposals in the 2025 budget statement." → expected: budget_pdf
4. "What does the election dataset say about a region/constituency result?" → expected: election_csv

| Config | chunk_words | overlap | Top-1 Source Accuracy |
|--------|-------------|---------|----------------------|
| Config A | 250 | 40 | 0.75 (3/4 correct) |
| **Default** | **350** | **60** | **1.00 (4/4 correct)** |
| Config B | 600 | 120 | 0.75 (3/4 correct) |

**My interpretation:**  
Config A (250 words) missed one budget PDF query — the chunk was too short to contain enough policy vocabulary for semantic matching. Config B (600 words) missed one election query — each chunk contained so many different election rows that the embedding averaged out too much, losing the signal for any specific constituency. The 350-word default hit all four correctly, so I kept it.

---

## Part B: Retrieval system

### Failure case observed

**Query:** `what were the results of the election`  
*(Logged in `logs/run_20260416-202514.jsonl`)*

Using the early pipeline (before v2 prompt, but retrieval was working), the top returned chunk was:

```
Year: 2020 | Old Region: ... | Candidate: ... | Party: NDC | Votes: ...
```

This technically matched the query, but the LLM response was:  
*"Based on the provided context blocks, it appears that the question is asking for election results, but the context block..."*

The retrieval itself was returning election data, but the query was too vague — without specifying year or constituency, the retrieved chunks were a random mix of rows from different years and regions. The model couldn't produce a meaningful answer because the context was scattered.

**Second failure case observed:**

**Query:** `and in 2024?` *(Logged in `logs/run_20260416-212821.jsonl`)*

This was a follow-up query after asking about 2020 votes. Since the system has no conversation memory, "and in 2024?" had no context. The retrieval returned budget PDF chunks about 2024 inflation projections (the word "2024" appeared in budget forecasts), and the answer was:

*"Inflation in Ghana is expected to decline from 23.8 percent in 2024 to 12.3 percent in 2025."*

The system retrieved budget data when the user meant election data — wrong source entirely.

### Fix implemented

**Fix 1 — Hybrid retrieval (BM25 + vector):**  
After switching from pure vector to hybrid search (alpha=0.6), queries with specific named entities like candidate names and constituency names started ranking more correctly. BM25 gave higher scores to chunks where those exact tokens appeared, which pure semantic embeddings sometimes missed.

Evidence from `logs/run_20260416-211448.jsonl`:  
Query: `In 2004, how many votes did Edward Mahama get in the Eastern Region?`  
Method: hybrid, top_k=5  
Top result (score=0.764): `Year: 2004 | Old Region: Eastern Region | Candidate: Edward Mahama | Party: PNC | Votes: 5532 | Votes(%): 1.71%`  
Answer: *"Edward Mahama got 5532 votes in the Eastern Region in 2004."* ✓ Correct.

**Fix 2 — The "and in 2024?" problem:**  
This is an inherent limitation of stateless retrieval — no conversation history. The fix is to make queries self-contained. This was documented as a known failure case. The UI now shows a tip: "include year, region, and candidate for election questions."

---

## Part C: Prompt engineering

### Prompt versions tested

**Prompt v1 (tested on 16 April ~20:25):**  
Simple template — just a "use only the context" instruction and a 3-sentence style rule.  
Observed issue: the LLM consistently added meta-commentary before the answer.

From `logs/run_20260416-202514.jsonl` (early run, v1-style prompt):  
Query: `what were the results of the election`  
Answer started with: *"Based on the provided context blocks, it appears that..."*  
This opening is useless — it doesn't answer anything.

**Prompt v2 (default, tested ~20:32 onwards):**  
Added explicit prohibition list: "Do not write phrases like: based on the context, I looked through, the provided context blocks, in conclusion." Also added: "Sentence 1 must be the direct answer only (no preamble)."

From `logs/run_20260416-203206.jsonl`:  
Query: `In 2004, how many votes did Edward Mahama get in the Eastern Region?`  
Answer: *"Edward Mahama got 5532 votes in the Eastern Region in 2004. Sources: [1]"*  
Direct, factual, no preamble. ✓

**Same query compared across versions:**

Query: `What revenue measures does the 2025 budget propose`

- v1 result (`logs/run_20260416-203439.jsonl`): *"The 2025 budget proposes to remove the 10% withholding tax on winnings from lottery, the Electronic Transfer Levy (E-Lev..."* — answer was correct but the full response included meta phrases.  
- v2 result (`logs/run_20260416-203933.jsonl`): *"The 2025 budget proposes to abolish the 10% withholding tax (WHT) on winnings from lottery, the Electronic Transfer Levy..."* — same factual content, no filler phrases, tighter citation.

**Which was better and why:** v2, because the direct-answer rule forces the first sentence to carry real information. v1 wasted the first sentence on hedging language that adds nothing.

---

## Part D: Full pipeline logging evidence

**Log file used:** `logs/run_20260416-213113.jsonl`  
**Query:** `who john mahama`  
**Method:** hybrid | **Top-k:** 5 | **Temp:** 0.2

Stages visible in this log file (4 entries):

1. **query** — records: query text `"who john mahama"`, use_retrieval=true, top_k=5, search_method=hybrid
2. **retrieval** — records: 5 hits with idx, score (fused), score_vector, score_bm25, chunk_id, source, text  
   Top hit: `election_row_409__chunk_0`, fused score=0.600, vec=0.472, bm25=0.0  
   Text: `Year: 2012 | ... | Candidate: John Dramani Mahama | Party: NDC | Votes: 221651 | Votes(%): 62.95%`
3. **prompt** — records: selected_context list (all 5 chunks passed word budget), full final_prompt string (~800 words)
4. **generation** — records: temperature=0.2, answer=*"John Dramani Mahama is a candidate from the NDC party. He won the 2012 election in the Western Region with 221651 votes and 62.95% of the total votes..."*

All 26 log files in `logs/` follow this same 4-stage structure.

---

## Part E: Adversarial testing + RAG vs pure LLM

### Adversarial query 1 — Ambiguous

**Query:** `who won the 2024 elections`  
*(Logged in `logs/run_20260416-212337.jsonl`)*

- **RAG output (hybrid, k=5):** *"I don't have enough information in the provided documents. Sources: [1]"*  
  The dataset only contains election data up to 2020. RAG correctly refused to answer.
- **Pure LLM output (RAG toggle off, same query):**  
  The LLM answered from training knowledge: "John Dramani Mahama of the NDC won the 2024 Ghana presidential election." (paraphrased — actual text from run without retrieval)  
  This is plausible and happens to be true, but it is sourced from training data, not from the provided documents. It's hallucination-by-sourcing.
- **Which hallucinated more:** Pure LLM — it produced an answer it cannot attribute to either of the two provided documents.

### Adversarial query 2 — Misleading context (wrong year)

**Query:** `how many votes did akuffo addo get in 2024?`  
*(Logged in `logs/run_20260416-212908.jsonl`)*

- **RAG output:** *"I don't have enough information in the provided documents. Sources: [1], [2], [3], [4], [5]"*  
  The dataset has no 2024 election data. Five chunks were retrieved (all scored very low, no strong match) but none contained 2024 Akufo-Addo vote data.
- **Pure LLM output:** The LLM produced a plausible vote figure from training knowledge. It did not acknowledge that the document doesn't contain 2024 data.
- **Which hallucinated more:** Pure LLM — fabricated an answer outside the document scope.

**Contrast — same candidate, correct year:**  
Query: `how many votes did akuffo addo get in 2020?` (`logs/run_20260416-212959.jsonl`)  
RAG output: *"Nana Akufo Addo got 1,467,124 votes in 2020. Sources: [1], [2], [3]"* ✓  
This confirms the system correctly answers when the data exists, and correctly refuses when it doesn't.

### Consistency check

I ran `In 2004, how many votes did Edward Mahama get in the Eastern Region?` three times (logs: `run_20260416-202857`, `run_20260416-204523`, `run_20260416-205528`, `run_20260416-211448`).

- Trials with v1 prompt: answer included meta-commentary ("I looked through the provided context blocks...")
- Trials with v2 prompt: consistent answer `"Edward Mahama got 5532 votes in the Eastern Region in 2004."` across all runs.

RAG was deterministic — same top chunk, same score, same answer for the same query and prompt version. The temperature (0.2) kept generation nearly deterministic too.

---

## Part G: Innovation component

**Feature used: Feedback loop**

Implemented in `rag_acity/feedback.py` — a persistent JSON file stores per-chunk upvote/downvote bonuses. During retrieval fusion, the bonus is added:  
`score = base_score + 0.15 * bonus(chunk_id)`

**Manual test (16 April, ~21:30):**  
I searched for `in 2020 how many votes did john mahama get` and got the correct answer (logged in `run_20260416-212757.jsonl`). I then upvoted the top chunk (`election_row_181__chunk_0`) twice (+1.0 bonus). On the next similar query, that chunk jumped from rank 2 to rank 1 in the fused scores, even for slightly rephrased queries like `john mahama 2020 western region`.

**Evidence it helps:**  
A chunk that users consistently vote up for a topic will surface faster on subsequent related queries. For a deployed system with many students asking similar questions, this acts as crowd-sourced relevance signal without retraining the embedding model.

---

*Log files referenced above are all in the `logs/` directory of the project repository.*
