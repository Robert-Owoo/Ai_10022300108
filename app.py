"""
Student Name: Robert George Owoo
Index Number: 10022300108

Streamlit UI for Manual RAG Chatbot — Academic City Edition.
Shows:
- Retrieved chunks with visual score bars and source badges
- Similarity scores (vector + BM25 + fused)
- Final prompt sent to LLM
- RAG pipeline diagram
"""

from __future__ import annotations

import os

import streamlit as st
from openai import APIError, AuthenticationError

from rag_acity.config import AppConfig
from rag_acity.env_bootstrap import load_dotenv_files, openai_api_key_configured, openai_env_warnings
from rag_acity.feedback import FeedbackStore
from rag_acity.greetings import greeting_reply, is_greeting_only
from rag_acity.logging_utils import StageLogger
from rag_acity.pipeline import load_index, run_rag


def _hydrate_openai_from_streamlit_secrets() -> None:
    try:
        sec = st.secrets
    except Exception:
        return
    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        try:
            if k not in sec:
                continue
            val = str(sec[k]).strip()
            if val and not os.environ.get(k, "").strip():
                os.environ[k] = val
        except Exception:
            continue


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
            }

            :root {
                --bg:         #080f0a;
                --bg2:        #0d1a10;
                --surface:    rgba(15, 30, 18, 0.80);
                --border:     rgba(212, 160, 23, 0.22);
                --gold:       #d4a017;
                --gold-dim:   rgba(212, 160, 23, 0.14);
                --gold-glow:  rgba(212, 160, 23, 0.08);
                --election:   #10b981;
                --election-d: rgba(16, 185, 129, 0.15);
                --budget:     #818cf8;
                --budget-d:   rgba(129, 140, 248, 0.15);
                --text:       #e8f0ea;
                --muted:      #7a9480;
                --danger:     #f87171;
            }

            /* ── Background ── */
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(ellipse 1000px 600px at 0% 0%,   rgba(16,185,129,0.06), transparent 55%),
                    radial-gradient(ellipse  800px 500px at 100% 30%, rgba(212,160,23,0.07), transparent 50%),
                    radial-gradient(ellipse  600px 400px at 50% 90%,  rgba(129,140,248,0.05), transparent 55%),
                    linear-gradient(160deg, #080f0a 0%, #0c1a0e 50%, #080d0a 100%);
            }

            [data-testid="stHeader"] { background: transparent; }

            /* ── Sidebar ── */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0a1a0d 0%, #070f09 100%) !important;
                border-right: 1px solid var(--border) !important;
                box-shadow: 4px 0 40px rgba(0,0,0,0.5);
            }
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span:not([style*="color: rgb"]) {
                color: var(--text) !important;
            }
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: var(--gold) !important;
                font-weight: 700 !important;
                letter-spacing: -0.01em;
            }

            /* ── Tabs ── */
            [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
                background-color: var(--gold) !important;
            }
            [data-testid="stTabs"] button { color: var(--muted) !important; font-weight: 500; }
            [data-testid="stTabs"] [aria-selected="true"] { color: var(--text) !important; }

            /* ── Expanders ── */
            div[data-testid="stExpander"] details {
                border: 1px solid var(--border) !important;
                border-radius: 10px;
                background: var(--surface);
            }

            /* ── Chat messages ── */
            [data-testid="stChatMessage"] {
                border: 1px solid rgba(212,160,23,0.12);
                border-radius: 14px;
                margin-bottom: 0.6rem;
                background: var(--surface);
                backdrop-filter: blur(8px);
            }

            /* ── Hero header ── */
            .ac-header {
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 0.8rem 0 1rem 0;
                border-bottom: 1px solid var(--border);
                margin-bottom: 1rem;
            }
            .ac-logo-ring {
                width: 48px; height: 48px;
                border-radius: 12px;
                background: linear-gradient(135deg, #0d5c2e 0%, #1a8c48 100%);
                border: 2px solid var(--gold);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.5rem;
                flex-shrink: 0;
                box-shadow: 0 0 20px rgba(212,160,23,0.25);
            }
            .ac-title { flex: 1; }
            .ac-title h1 {
                font-size: 1.55rem; font-weight: 700;
                letter-spacing: -0.03em; margin: 0 0 0.2rem 0;
                background: linear-gradient(90deg, #e8f0ea 0%, var(--gold) 60%, #e8f0ea 100%);
                background-size: 200%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .ac-title p { margin: 0; color: var(--muted); font-size: 0.88rem; }

            /* ── Badges ── */
            .badge {
                display: inline-flex; align-items: center; gap: 0.3rem;
                font-size: 0.7rem; font-weight: 600;
                letter-spacing: 0.05em; text-transform: uppercase;
                padding: 0.18rem 0.55rem;
                border-radius: 999px;
            }
            .badge-gold    { background: var(--gold-dim);     color: var(--gold);     border: 1px solid rgba(212,160,23,0.35); }
            .badge-election{ background: var(--election-d);   color: var(--election); border: 1px solid rgba(16,185,129,0.35); }
            .badge-budget  { background: var(--budget-d);     color: var(--budget);   border: 1px solid rgba(129,140,248,0.35); }
            .badge-muted   { background: rgba(122,148,128,0.1); color: var(--muted);  border: 1px solid rgba(122,148,128,0.25); }

            /* ── Score bar ── */
            .score-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; }
            .score-label { font-size: 0.72rem; color: var(--muted); width: 52px; flex-shrink: 0; }
            .score-track {
                flex: 1; height: 6px; border-radius: 999px;
                background: rgba(255,255,255,0.07);
                overflow: hidden;
            }
            .score-fill-fused    { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--gold), #f5c842); }
            .score-fill-vector   { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--budget), #a5b4fc); }
            .score-fill-bm25     { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--election), #34d399); }
            .score-value { font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: var(--text); width: 38px; text-align: right; flex-shrink: 0; }

            /* ── Chunk card ── */
            .chunk-card {
                border-radius: 10px;
                border: 1px solid var(--border);
                background: var(--surface);
                padding: 0.75rem 1rem;
                margin-bottom: 0.6rem;
            }
            .chunk-meta {
                display: flex; align-items: center; gap: 0.5rem;
                margin-bottom: 0.5rem; flex-wrap: wrap;
            }
            .chunk-rank {
                font-size: 0.78rem; font-weight: 700;
                color: var(--gold); width: 22px;
            }
            .chunk-text {
                font-size: 0.85rem; color: var(--text);
                line-height: 1.55;
                border-left: 2px solid var(--border);
                padding-left: 0.6rem;
                margin: 0.5rem 0;
            }

            /* ── Suggested questions ── */
            .suggestion-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.5rem;
                margin: 1rem 0;
            }
            .suggestion-card {
                padding: 0.65rem 0.85rem;
                border-radius: 10px;
                border: 1px solid var(--border);
                background: var(--gold-glow);
                font-size: 0.82rem;
                color: var(--text);
                cursor: pointer;
                transition: border-color 0.15s;
            }
            .suggestion-card:hover { border-color: var(--gold); }
            .suggestion-icon { font-size: 1rem; display: block; margin-bottom: 0.25rem; }

            /* ── Pipeline diagram ── */
            .pipeline-step {
                display: flex; align-items: flex-start; gap: 0.75rem;
                padding: 0.75rem 1rem;
                border-radius: 10px;
                border: 1px solid var(--border);
                background: var(--surface);
                margin-bottom: 0.5rem;
            }
            .pipeline-icon {
                font-size: 1.4rem; flex-shrink: 0;
                width: 36px; height: 36px;
                display: flex; align-items: center; justify-content: center;
            }
            .pipeline-content h4 { margin: 0 0 0.15rem 0; font-size: 0.9rem; color: var(--gold); font-weight: 600; }
            .pipeline-content p  { margin: 0; font-size: 0.8rem; color: var(--muted); }
            .pipeline-arrow { text-align: center; color: var(--border); font-size: 1.1rem; margin: -0.1rem 0; }

            /* ── Stat card ── */
            .stat-row {
                display: flex; gap: 0.5rem; margin: 0.75rem 0;
            }
            .stat-card {
                flex: 1; padding: 0.6rem 0.75rem;
                border-radius: 8px;
                border: 1px solid var(--border);
                background: var(--gold-glow);
                text-align: center;
            }
            .stat-value { font-size: 1.1rem; font-weight: 700; color: var(--gold); display: block; }
            .stat-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }

            /* ── Slider accent ── */
            [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
                background-color: var(--gold) !important;
                border: 2px solid #fff2 !important;
            }

            /* ── Metrics ── */
            [data-testid="stMetric"] { background: var(--gold-glow); border-radius: 8px; padding: 0.5rem; border: 1px solid var(--border); }
            [data-testid="stMetricValue"] { color: var(--gold) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Source helpers ────────────────────────────────────────────────────────────
def _source_badge(source: str) -> str:
    if source == "election_csv":
        return '<span class="badge badge-election">🗳 Election CSV</span>'
    if source == "budget_pdf":
        return '<span class="badge badge-budget">📊 Budget PDF</span>'
    return f'<span class="badge badge-muted">{source}</span>'


def _score_bars(fused: float, vec: float, bm25: float) -> str:
    def bar(label: str, val: float, cls: str) -> str:
        pct = max(0.0, min(1.0, val)) * 100
        return (
            f'<div class="score-row">'
            f'<span class="score-label">{label}</span>'
            f'<div class="score-track"><div class="{cls}" style="width:{pct:.1f}%"></div></div>'
            f'<span class="score-value">{val:.3f}</span>'
            f'</div>'
        )
    return bar("Fused", fused, "score-fill-fused") + bar("Vector", vec, "score-fill-vector") + bar("BM25", bm25, "score-fill-bm25")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACity RAG · Robert George Owoo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv_files()
_hydrate_openai_from_streamlit_secrets()
_inject_styles()

cfg = AppConfig()

if "chat" not in st.session_state:
    st.session_state.chat = []


def _clear_chat() -> None:
    st.session_state.chat = []


def index_ready() -> bool:
    return cfg.embeddings_path().exists() and cfg.metadata_path().exists() and cfg.bm25_path().exists()


if not index_ready():
    st.error(
        "Index not found. From the project folder run:\n\n"
        "`.venv/Scripts/python -m scripts.build_index`"
    )
    st.stop()


@st.cache_resource
def _load_index_cached():
    return load_index(cfg)


index = _load_index_cached()
logger = StageLogger(cfg.logs_dir)
feedback_store = FeedbackStore(cfg.feedback_path())

# Knowledge-base stats
_n_chunks = len(index.vector_store.metadata)
_n_election = sum(1 for m in index.vector_store.metadata if m.get("source") == "election_csv")
_n_budget   = _n_chunks - _n_election

_search_labels = {
    "hybrid": "⚡ Hybrid (vector + keyword)",
    "vector": "🔮 Vector only",
    "bm25":   "🔑 Keyword (BM25) only",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0 1rem">'
        '<div style="font-size:2.2rem">🎓</div>'
        '<div style="font-weight:700;font-size:1rem;color:#d4a017;letter-spacing:-0.01em">Academic City RAG</div>'
        '<div style="font-size:0.72rem;color:#7a9480;margin-top:0.2rem">Robert George Owoo · 10022300108</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-card"><span class="stat-value">{_n_chunks}</span><span class="stat-label">Chunks</span></div>'
        f'<div class="stat-card"><span class="stat-value" style="color:#10b981">{_n_election}</span><span class="stat-label">Election</span></div>'
        f'<div class="stat-card"><span class="stat-value" style="color:#818cf8">{_n_budget}</span><span class="stat-label">Budget</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### ⚙️ Controls")

    if st.button("＋ New chat", type="primary", use_container_width=True):
        _clear_chat()
        st.rerun()

    use_retrieval = st.toggle("🔍 RAG Retrieval", value=True,
                              help="Disable to compare pure LLM (no document lookup).")
    top_k = st.slider("Chunks to retrieve (k)", min_value=3, max_value=10, value=5)
    search_method = st.selectbox(
        "Search method",
        ["hybrid", "vector", "bm25"],
        index=0,
        format_func=lambda m: _search_labels.get(m, m),
    )
    prompt_version = st.selectbox(
        "Prompt template",
        ["v2", "v2-verbose", "v1"],
        index=0,
        help="v2: concise grounded. v2-verbose: detailed. v1: simple baseline.",
    )

    with st.expander("🔬 Advanced", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
                                help="Low = factual and consistent. High = more creative.")
        st.caption("Keep ≤ 0.35 for factual RAG answers.")

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    st.markdown(
        '<span class="badge badge-election">🗳 Election CSV</span> Ghana presidential results (multi-year)<br><br>'
        '<span class="badge badge-budget">📊 Budget PDF</span> MOFEP 2025 Budget Statement',
        unsafe_allow_html=True,
    )
    st.caption("Manual pipeline — no LangChain / LlamaIndex.")

    st.divider()
    with st.expander("🔌 API Connection", expanded=False):
        load_dotenv_files()
        _hydrate_openai_from_streamlit_secrets()
        _key = os.environ.get("OPENAI_API_KEY", "").strip()
        status = "✅ Connected" if _key else "❌ Missing key"
        st.markdown(f"**Status:** {status}")
        for hint in openai_env_warnings():
            st.warning(hint)
        st.code(
            "MODEL: " + os.environ.get("OPENAI_MODEL", "(default)") + "\n"
            "BASE:  " + (os.environ.get("OPENAI_BASE_URL", "") or "(OpenAI default)"),
            language="text",
        )

    if st.button("🗑 Clear conversation", use_container_width=True):
        _clear_chat()
        st.rerun()


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ac-header">'
    '<div class="ac-logo-ring">🎓</div>'
    '<div class="ac-title">'
    '<h1>Academic City RAG Assistant</h1>'
    '<p>CS4241 · Manual RAG · Ghana Election Results &amp; 2025 Budget Statement &nbsp;'
    '<span class="badge badge-gold">Manual RAG</span> '
    '<span class="badge badge-muted">No LangChain</span>'
    '</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_debug, tab_pipeline = st.tabs(["💬 Conversation", "🔍 Sources & Scores", "⚙️ Pipeline"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONVERSATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Pick up any suggestion that was clicked on the previous render
    pending_query = st.session_state.pop("_pending_query", None)

    if not st.session_state.chat:
        # Suggested questions when chat is empty
        st.markdown(
            '<p style="color:#7a9480;font-size:0.88rem;margin-bottom:0.75rem">'
            'Try one of these questions to get started:</p>',
            unsafe_allow_html=True,
        )
        suggestions = [
            ("🗳", "In 2004, how many votes did Edward Mahama get in the Eastern Region?"),
            ("📊", "What revenue measures does the 2025 budget propose?"),
            ("🗳", "How many votes did John Mahama get in the 2020 election?"),
            ("📊", "What is Ghana's projected inflation rate for 2025?"),
            ("🗳", "Who won the 2016 election in the Northern Region?"),
            ("📊", "What does the 2025 budget say about the E-Levy?"),
        ]
        cols = st.columns(2)
        for idx, (icon, q) in enumerate(suggestions):
            with cols[idx % 2]:
                if st.button(f"{icon}  {q}", key=f"sug_{idx}", use_container_width=True):
                    st.session_state._pending_query = q
                    st.rerun()
    else:
        st.caption("💡 Tip: include year, region, and candidate for election questions. Use **RAG toggle** in sidebar to compare pure LLM.")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    typed_prompt = st.chat_input("Ask about the election results or the 2025 budget…")
    if prompt := (pending_query or typed_prompt):
        load_dotenv_files()
        _hydrate_openai_from_streamlit_secrets()
        st.session_state.chat.append({"role": "user", "content": prompt})

        if is_greeting_only(prompt):
            st.session_state.chat.append({
                "role": "assistant",
                "content": greeting_reply(prompt),
                "meta": {"retrieved": [], "selected_context": [], "final_prompt": "",
                         "answer": "", "log_path": None, "retrieval_settings": {"note": "greeting"}},
            })
            st.rerun()

        if not openai_api_key_configured():
            st.session_state.chat.append({
                "role": "assistant",
                "content": "**API key missing.** Add `OPENAI_API_KEY` to `.env` then rerun.",
                "meta": {"retrieved": [], "selected_context": [], "final_prompt": "",
                         "answer": "", "log_path": None, "retrieval_settings": {}},
            })
            st.rerun()

        try:
            with st.spinner("🔍 Retrieving → 📝 Building context → 🤖 Generating…"):
                res = run_rag(
                    cfg=cfg, index=index, query=prompt,
                    prompt_version=prompt_version, temperature=temperature,
                    use_retrieval=use_retrieval, top_k=top_k,
                    search_method=search_method, logger=logger,
                )
        except RuntimeError as e:
            res = {"answer": f"**Config error:** {e}", "retrieved": [], "selected_context": [],
                   "final_prompt": "", "log_path": None, "retrieval_settings": {}}
        except AuthenticationError as e:
            res = {"answer": f"**Auth failed (401):** `{e}`\n\nCheck your API key in `.env`.",
                   "retrieved": [], "selected_context": [], "final_prompt": "",
                   "log_path": None, "retrieval_settings": {}}
        except APIError as e:
            code = getattr(e, "status_code", None)
            res = {"answer": f"**API error ({code}):** `{e}`", "retrieved": [], "selected_context": [],
                   "final_prompt": "", "log_path": None, "retrieval_settings": {}}
        except Exception as e:
            res = {"answer": f"**Error:** `{type(e).__name__}` — {e}", "retrieved": [],
                   "selected_context": [], "final_prompt": "", "log_path": None, "retrieval_settings": {}}

        st.session_state.chat.append({"role": "assistant", "content": res["answer"], "meta": res})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SOURCES & SCORES
# ══════════════════════════════════════════════════════════════════════════════
with tab_debug:
    if not (st.session_state.chat and st.session_state.chat[-1]["role"] == "assistant"):
        st.markdown(
            '<div style="text-align:center;padding:3rem 1rem;color:#7a9480">'
            '<div style="font-size:2.5rem">🔍</div>'
            '<p>Send a message to see retrieved chunks, scores, and the full prompt here.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        meta = st.session_state.chat[-1].get("meta", {})
        rs   = meta.get("retrieval_settings") or {}

        # ── Run metrics ───────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Retrieved (k)", len(meta.get("retrieved", []) or []))
        with c2: st.metric("Top-k setting", rs.get("top_k", "—"))
        with c3: st.metric("Search mode",   str(rs.get("search_method", "—")))
        with c4: st.metric("Prompt ver.",   str(prompt_version))
        st.caption(f"Temperature `{temperature:.2f}` &nbsp;·&nbsp; Log `{meta.get('log_path', '—')}`")

        st.divider()

        # ── Retrieved chunks ──────────────────────────────────────────────────
        retrieved = meta.get("retrieved", [])
        if not retrieved:
            st.info("No retrieval (pure LLM mode) or no hits for this query.")
        else:
            st.markdown(f"#### Retrieved Chunks &nbsp; <span class='badge badge-gold'>{len(retrieved)} chunks</span>", unsafe_allow_html=True)
            for i, h in enumerate(retrieved, start=1):
                source   = h.get("source", "")
                chunk_id = str(h.get("chunk_id", ""))
                fused    = float(h.get("score", 0))
                vec      = float(h.get("score_vector", 0))
                bm25_s   = float(h.get("score_bm25", 0))
                current  = feedback_store.bonus(chunk_id) if chunk_id else 0.0

                with st.expander(
                    f"#{i} · Fused {fused:.3f} · {h.get('source','')}", expanded=(i == 1)
                ):
                    # Header row
                    st.markdown(
                        f'<div class="chunk-meta">'
                        f'<span class="chunk-rank">#{i}</span>'
                        f'{_source_badge(source)}'
                        f'<span class="badge badge-muted" style="font-family:monospace">{chunk_id}</span>'
                        f'{"<span class=\'badge badge-gold\'>⭐ top</span>" if i == 1 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    # Score bars
                    st.markdown(_score_bars(fused, vec, bm25_s), unsafe_allow_html=True)
                    st.markdown("---")
                    # Chunk text
                    st.markdown(
                        f'<div class="chunk-text">{h.get("text", "")}</div>',
                        unsafe_allow_html=True,
                    )
                    # Feedback
                    st.markdown("**Relevance feedback** — vote to adjust future rankings:")
                    bonus_color = "#10b981" if current > 0 else ("#f87171" if current < 0 else "#7a9480")
                    st.markdown(
                        f'<span style="font-size:0.8rem;color:{bonus_color}">Current bonus: {current:+.2f}</span>',
                        unsafe_allow_html=True,
                    )
                    b1, b2, b3 = st.columns([1, 1, 2])
                    with b1:
                        if st.button("👍 Up", key=f"up_{chunk_id}_{i}"):
                            feedback_store.update(chunk_id, delta=0.5)
                            st.rerun()
                    with b2:
                        if st.button("👎 Down", key=f"down_{chunk_id}_{i}"):
                            feedback_store.update(chunk_id, delta=-0.5)
                            st.rerun()
                    with b3:
                        if st.button("↺ Reset", key=f"reset_{chunk_id}_{i}"):
                            table = feedback_store.load()
                            table.pop(chunk_id, None)
                            feedback_store.save(table)
                            st.rerun()

        st.divider()
        st.markdown("#### Full Prompt Sent to LLM")
        with st.expander("Show full prompt", expanded=False):
            st.code(meta.get("final_prompt", ""), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PIPELINE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("### RAG Pipeline Architecture")
    st.caption("End-to-end flow for every query — all components implemented manually.")

    steps = [
        ("💬", "User Query", "Natural language question entered in the chat interface."),
        ("🔢", "Query Embedding", f"Encoded with `{cfg.embedding_model_name}` → 384-dim normalized vector."),
        ("⚡", "Hybrid Retrieval", f"Dense cosine similarity (α={cfg.hybrid_alpha}) + BM25 keyword score — fused via min-max normalization. Mode: **{_search_labels.get('hybrid','')}**."),
        ("⭐", "Feedback Reranking", "User upvote/downvote bonuses applied to fused scores (weight=0.15). Innovation component (Part G)."),
        ("✂️", "Context Selection", f"Top-k chunks filtered by {cfg.max_context_words}-word budget — greedy selection by rank."),
        ("📝", "Prompt Construction", "Retrieved chunks injected into structured template with hallucination-control rules (v1 / v2 / v2-verbose)."),
        ("🤖", "LLM Generation", f"Sent to `{os.environ.get('OPENAI_MODEL','llm')}` at temperature {0.2:.2f}. Answer grounded strictly in provided context."),
        ("📋", "Response + Logging", "Answer returned to UI. All stages (query → retrieval → prompt → generation) written to JSONL log."),
    ]

    for i, (icon, title, desc) in enumerate(steps):
        st.markdown(
            f'<div class="pipeline-step">'
            f'<div class="pipeline-icon">{icon}</div>'
            f'<div class="pipeline-content">'
            f'<h4>{title}</h4>'
            f'<p>{desc}</p>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if i < len(steps) - 1:
            st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Knowledge Base")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="chunk-card">'
            f'<div class="chunk-meta">{_source_badge("election_csv")}</div>'
            f'<span class="stat-value" style="color:#10b981;font-size:1.4rem">{_n_election}</span>'
            f'<span class="stat-label" style="font-size:0.78rem;color:#7a9480"> chunks</span>'
            f'<p style="font-size:0.78rem;color:#7a9480;margin-top:0.4rem">Ghana presidential election results — multi-year CSV records (row-per-constituency).</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="chunk-card">'
            f'<div class="chunk-meta">{_source_badge("budget_pdf")}</div>'
            f'<span class="stat-value" style="color:#818cf8;font-size:1.4rem">{_n_budget}</span>'
            f'<span class="stat-label" style="font-size:0.78rem;color:#7a9480"> chunks</span>'
            f'<p style="font-size:0.78rem;color:#7a9480;margin-top:0.4rem">MOFEP 2025 Budget Statement PDF — policy narrative, fiscal targets, sector allocations.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="chunk-card">'
            f'<div class="chunk-meta"><span class="badge badge-gold">⚙️ Config</span></div>'
            f'<p style="font-size:0.8rem;color:#7a9480;margin:0.3rem 0">'
            f'Chunk size: <b style="color:#e8f0ea">{cfg.chunk_words} words</b><br>'
            f'Overlap: <b style="color:#e8f0ea">{cfg.chunk_overlap_words} words</b><br>'
            f'Embedding: <b style="color:#e8f0ea">all-MiniLM-L6-v2</b><br>'
            f'Vector dims: <b style="color:#e8f0ea">384</b><br>'
            f'Hybrid α: <b style="color:#e8f0ea">{cfg.hybrid_alpha}</b>'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
