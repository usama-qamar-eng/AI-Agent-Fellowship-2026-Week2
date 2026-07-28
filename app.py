import time
import pandas as pd
import streamlit as st

from config import (
    APP_PASSWORD, CHUNK_OVERLAP, CHUNK_SIZE_OPTIONS, DEFAULT_CHUNK_SIZE, DEFAULT_PROMPT_TEMPLATE,
    LLM_MODELS, PROMPT_TEMPLATES, SUPPORTED_EXTENSIONS,
)
from core.chunker import chunk_pages
from core.loader import load_pages
from core.llm import summarize_and_suggest
from core.pipeline import answer_question, compare_documents
from core.store import Store
from experiments.common import TEST_QUESTION, boundary_overlap_chars, chunk_by_file, pick_relevant_and_irrelevant

st.set_page_config(page_title="Document Intelligence", page_icon="📚", layout="wide")

@st.cache_resource(show_spinner=False)
def get_store() -> Store:
    return Store()  # loads the embedding model + opens ChromaDB once per server process

@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def init_state() -> None:
    st.session_state.setdefault("chat_history", [])       # [{"role","content"}]
    st.session_state.setdefault("last_sources", [])        # sources for the last answer
    st.session_state.setdefault("doc_filter", "All documents")
    st.session_state.setdefault("total_tokens", 0)
    st.session_state.setdefault("tokens_by_model", {})     # {model_id: tokens} — needed for cost estimate
    st.session_state.setdefault("raw_files", {})           # {filename: bytes} — enables real refresh
    st.session_state.setdefault("summaries", {})           # {filename: summary}
    st.session_state.setdefault("suggested_qs", {})        # {filename: [questions]}
    st.session_state.setdefault("prefill_question", "")
    st.session_state.setdefault("authenticated", not APP_PASSWORD)  # no password set -> skip login entirely

def check_auth() -> bool:
    """Real password check against APP_PASSWORD (env-configured). Returns True if the
    app should render normally; False if it already rendered a login screen and should stop."""
    if st.session_state.authenticated:
        return True

    st.title("🔒 Document Intelligence — Login")
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Log in", type="primary", key="login_btn"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password.")
    return False

def track_tokens(model: str, tokens: int) -> None:
    st.session_state.total_tokens += tokens
    st.session_state.tokens_by_model[model] = st.session_state.tokens_by_model.get(model, 0) + tokens

def process_upload(store: Store, model: str, chunk_size: int, name: str, file_bytes: bytes) -> None:
    pages = load_pages(name, file_bytes)
    chunks = chunk_pages(pages, chunk_size, CHUNK_OVERLAP)
    store.delete_document(name)  # safe no-op if it wasn't there — avoids duplicate chunks on re-upload
    if chunks:
        store.add(name, chunks)
    st.session_state.raw_files[name] = file_bytes
    st.session_state[f"chunk_size_used_{name}"] = chunk_size  # so Refresh can reuse the same size later
    st.session_state[f"status_{name}"] = (len(pages), len(chunks))

    full_text = " ".join(t for _, t in pages)
    if full_text.strip():
        try:
            summary, questions, tokens = summarize_and_suggest(model, name, full_text)
            st.session_state.summaries[name] = summary
            st.session_state.suggested_qs[name] = questions
            track_tokens(model, tokens)
        except Exception as exc:
            st.session_state.summaries[name] = f"(Summary unavailable: {exc})"

def render_upload_tab(store: Store, model: str, chunk_size: int) -> None:
    st.subheader("📤 Upload Documents")
    st.caption(f"Using chunk size **{chunk_size}** chars.")
    files = st.file_uploader(
        "PDF, TXT, DOCX or MD — you can select several at once",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        key="file_uploader",
    )
    if files and st.button("Process documents", type="primary", key="process_docs_btn"):
        bar = st.progress(0.0)
        for i, file in enumerate(files):
            with st.spinner(f"Processing {file.name} (chunking, embedding, summarizing)…"):
                process_upload(store, model, chunk_size, file.name, file.read())
            bar.progress((i + 1) / len(files))
        st.success(f"Processed {len(files)} document(s).")

    for name in store.list_documents():
        key = f"status_{name}"
        if key not in st.session_state:
            continue
        pages, chunks = st.session_state[key]
        with st.expander(f"✅ **{name}** — {pages} page(s), {chunks} chunk(s) embedded", expanded=False):
            st.caption(st.session_state.summaries.get(name, "No summary generated yet."))
            for q in st.session_state.suggested_qs.get(name, []):
                if st.button(f"💬 {q}", key=f"sugg_{name}_{q}"):
                    st.session_state.prefill_question = q


def render_chat_tab(store: Store, model: str, prompt_template: str) -> None:
    st.subheader("💬 Chat with your documents")

    c1, c2, c3 = st.columns([2, 1, 1])
    docs = ["All documents"] + list(store.list_documents().keys())
    st.session_state.doc_filter = c1.selectbox("Search scope", docs, key="chat_doc_filter")
    use_hybrid = c2.checkbox("Hybrid search", help="Blend semantic similarity with keyword matching", key="chat_hybrid")
    if st.session_state.chat_history:
        transcript = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.chat_history)
        c3.download_button("⬇ Export", transcript, file_name="chat_export.md", mime="text/markdown")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "tokens" in msg:
                st.caption(f"⏱ {msg['elapsed']}s · 🪙 {msg['tokens']} tokens")
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} source chunk(s) retrieved"):
                        for s in msg["sources"]:
                            st.markdown(f"**{s['source']}**, page {s['page']}, chunk #{s['chunk_id']}")
                            st.text(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))

    doc_filter = None if st.session_state.doc_filter == "All documents" else st.session_state.doc_filter

    def _run(q: str) -> None:
        st.session_state.chat_history.append({"role": "user", "content": q})
        t0 = time.time()
        try:
            answer, sources, tokens = answer_question(
                store, model, q, st.session_state.chat_history[:-1], doc_filter, use_hybrid, prompt_template
            )
        except Exception as exc:
            answer, sources, tokens = f"❌ {exc}", [], 0
        elapsed = round(time.time() - t0, 2)
        track_tokens(model, tokens)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "elapsed": elapsed, "tokens": tokens, "sources": sources}
        )

    if st.session_state.prefill_question:
        st.info("💡 Suggested question loaded — edit if needed, then send.")
        edited = st.text_area("Edit before sending", value=st.session_state.prefill_question, height=80, key="prefill_edit")
        s1, s2 = st.columns([1, 4])
        if s1.button("Send ➤", type="primary", key="prefill_send"):
            st.session_state.prefill_question = ""
            _run(edited)
            st.rerun()
        if s2.button("✕ Cancel", key="prefill_cancel"):
            st.session_state.prefill_question = ""
            st.rerun()

    question = st.chat_input("Ask a question about your uploaded documents…", key="chat_input_box")
    if question:
        _run(question)
        st.rerun()

def render_library_tab(store: Store, model: str, chunk_size: int) -> None:
    st.subheader("📁 Document Library")
    docs = store.list_documents()
    if not docs:
        st.info("No documents uploaded yet.")
        return
    for name, count in docs.items():
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.markdown(f"**{name}** — {count} chunks")
        has_cache = name in st.session_state.raw_files
        if c2.button("🔄 Refresh", key=f"refresh_{name}", disabled=not has_cache,
                     help=None if has_cache else "Original file not cached this session — re-upload instead"):
            size = st.session_state.get(f"chunk_size_used_{name}", chunk_size)
            process_upload(store, model, size, name, st.session_state.raw_files[name])
            st.success(f"Re-embedded {name} (chunk size {size}).")
            st.rerun()
        if c3.button("🗑️ Delete", key=f"delete_{name}"):
            store.delete_document(name)
            st.session_state.raw_files.pop(name, None)
            st.rerun()

def render_compare_tab(store: Store, model: str, prompt_template: str) -> None:
    st.subheader("🆚 Compare Documents")
    docs = list(store.list_documents().keys())
    if len(docs) < 2:
        st.info("Upload at least 2 documents to compare them.")
        return
    c1, c2 = st.columns(2)
    doc_a = c1.selectbox("Document A", docs, index=0, key="compare_doc_a")
    doc_b = c2.selectbox("Document B", docs, index=1 if len(docs) > 1 else 0, key="compare_doc_b")
    question = st.text_input("Optional: focus the comparison on a specific question", key="compare_question")
    if doc_a == doc_b:
        st.warning("Pick two different documents.")
        return
    if st.button("Compare", type="primary", key="compare_run_btn"):
        with st.spinner("Retrieving relevant sections from both documents…"):
            try:
                answer, sources, tokens = compare_documents(store, model, doc_a, doc_b, question, prompt_template)
                track_tokens(model, tokens)
            except Exception as exc:
                answer, sources = f"❌ {exc}", []
        st.markdown(answer)
        if sources:
            with st.expander(f"📎 {len(sources)} source chunk(s) used"):
                for s in sources:
                    st.markdown(f"**{s['source']}**, page {s['page']}, chunk #{s['chunk_id']}")
                    st.text(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))


def render_stats_tab(store: Store) -> None:
    st.subheader("📊 Statistics")
    docs = store.list_documents()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(docs))
    c2.metric("Total chunks", sum(docs.values()))
    c3.metric("Chat turns", len(st.session_state.chat_history))
    c4.metric("Tokens used", f"{st.session_state.total_tokens:,}")

def _get_working_corpus() -> dict[str, list[tuple[int, str]]]:
    """Real corpus for the Experiments tab: whatever documents are uploaded this
    session, re-read from the cached raw bytes so it reflects the actual files. No
    bundled sample corpus — if nothing's uploaded yet, this returns empty and the
    caller shows a message asking for a real upload instead of substituting anything."""
    return {name: load_pages(name, data) for name, data in st.session_state.raw_files.items()}

def _render_chunk_size_experiment(corpus: dict) -> None:
    st.markdown(
        f"Compares **chunk_size = {', '.join(str(s) for s in CHUNK_SIZE_OPTIONS)}** "
        f"(overlap fixed at the project default, {CHUNK_OVERLAP}) — chunked live, right now, "
        "from the corpus above."
    )
    if st.button("▶ Compare chunk sizes", type="primary", key="run_exp_chunk_size"):
        pages = [p for file_pages in corpus.values() for p in file_pages]
        with st.spinner("Chunking at 3 sizes…"):
            rows = []
            for size in CHUNK_SIZE_OPTIONS:
                chunks = chunk_pages(pages, size=size, overlap=CHUNK_OVERLAP)
                lengths = [len(c["text"]) for c in chunks]
                rows.append({
                    "chunk_size": size, "chunks": len(chunks),
                    "avg_chunk_length": round(sum(lengths) / len(lengths)) if lengths else 0,
                })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Smaller chunks → more, more targeted pieces (better for pinpoint questions, risk "
            "splitting one idea in two). Larger chunks → fewer, broader pieces (more context per "
            "chunk, less targeted retrieval)."
        )

def _render_chunk_overlap_experiment(corpus: dict) -> None:
    st.markdown(
        f"Compares **chunk_overlap = 0, 50, 100** (chunk_size fixed at the project default, "
        f"{DEFAULT_CHUNK_SIZE}) — measuring the actual duplicated text at each chunk boundary, "
        "not a proxy."
    )
    if st.button("▶ Compare chunk overlaps", type="primary", key="run_exp_chunk_overlap"):
        pages = [p for file_pages in corpus.values() for p in file_pages]
        with st.spinner("Chunking at 3 overlaps…"):
            rows = []
            for overlap in (0, 50, 100):
                chunks = chunk_pages(pages, size=DEFAULT_CHUNK_SIZE, overlap=overlap)
                rows.append({
                    "overlap": overlap, "chunks": len(chunks),
                    "duplicated_boundary_chars": boundary_overlap_chars(chunks),
                })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "overlap=0 is cheapest but risks losing context exactly at a boundary. Higher overlap "
            "protects against that at the cost of more stored/embedded chunks."
        )

def _render_embedding_model_experiment(corpus: dict) -> None:
    st.markdown(
        "Picks the most and least keyword-relevant real chunk for your question (same "
        "technique as Hybrid Search), then compares real cosine similarity from two "
        "embedding models."
    )
    query = st.text_input("Test question", value=TEST_QUESTION, key="exp_embedding_question")
    if st.button("▶ Compare embedding models", type="primary", key="run_exp_embedding_models"):
        chunks_by_file = chunk_by_file(corpus, DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP)
        try:
            (rel_file, relevant), (irr_file, irrelevant) = pick_relevant_and_irrelevant(chunks_by_file, query)
        except ValueError as exc:
            st.warning(str(exc))
            return

        c1, c2 = st.columns(2)
        c1.markdown(f"**Most relevant** — {rel_file}")
        c1.text(relevant["text"][:300] + "…")
        c2.markdown(f"**Least relevant** — {irr_file}")
        c2.text(irrelevant["text"][:300] + "…")

        rows = []
        for model_name in ("all-MiniLM-L6-v2", "all-mpnet-base-v2"):
            with st.spinner(f"Loading {model_name} (first run downloads it — may take a minute)…"):
                try:
                    from sentence_transformers import util
                    embed_model = get_embedding_model(model_name)
                    embeddings = embed_model.encode(
                        [query, relevant["text"], irrelevant["text"]], convert_to_tensor=True
                    )
                    sim_r = util.cos_sim(embeddings[0], embeddings[1]).item()
                    sim_i = util.cos_sim(embeddings[0], embeddings[2]).item()
                    rows.append({
                        "model": model_name, "relevant_similarity": round(sim_r, 3),
                        "irrelevant_similarity": round(sim_i, 3), "gap": round(sim_r - sim_i, 3),
                    })
                except Exception as exc:
                    st.error(f"{model_name}: {exc}")

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                "A bigger gap means the model separates relevant from irrelevant content more "
                "sharply — usually better retrieval precision, at the cost of a larger, slower model."
            )

def render_experiments_tab() -> None:
    st.subheader("🧪 Experiments")
    corpus = _get_working_corpus()
    if not corpus:
        st.info("📤 Upload real document(s) in the Upload tab first — experiments run against "
                 "your actual uploaded content, not a bundled sample corpus.")
        return
    st.caption("Using your uploaded documents: " + ", ".join(corpus.keys()))
    exp_section = st.radio(
        "Experiment",
        ["📏 Chunk Size", "🔗 Chunk Overlap", "🧬 Embedding Models"],
        horizontal=True, key="exp_section", label_visibility="collapsed",
    )
    st.divider()

    if exp_section == "📏 Chunk Size":
        _render_chunk_size_experiment(corpus)
    elif exp_section == "🔗 Chunk Overlap":
        _render_chunk_overlap_experiment(corpus)
    elif exp_section == "🧬 Embedding Models":
        _render_embedding_model_experiment(corpus)

def main() -> None:
    init_state()
    if not check_auth():
        return  # login screen was just shown — stop here, nothing below this line has run yet

    with st.spinner("Loading knowledge base…"):
        store = get_store()

    with st.sidebar:
        st.markdown("## 📚 Document Intelligence")
        st.caption("Enterprise RAG · OpenRouter · ChromaDB")

        model_label = st.selectbox("LLM Model", list(LLM_MODELS.keys()), key="model_select")
        model = LLM_MODELS[model_label]

        chunk_size = st.selectbox(
            "Chunk size", CHUNK_SIZE_OPTIONS,
            index=CHUNK_SIZE_OPTIONS.index(DEFAULT_CHUNK_SIZE), key="chunk_size_select",
        )

        prompt_template = st.selectbox(
            "Prompt template", list(PROMPT_TEMPLATES.keys()),
            index=list(PROMPT_TEMPLATES.keys()).index(DEFAULT_PROMPT_TEMPLATE), key="prompt_template_select",
        )

        st.divider()
        if APP_PASSWORD and st.button("🔓 Log out", use_container_width=True, key="logout_btn"):
            st.session_state.authenticated = False
            st.rerun()
        if st.button("🗑️ Clear chat history", use_container_width=True, key="clear_history_btn"):
            st.session_state.chat_history = []
            st.rerun()

    st.title("Enterprise Document Intelligence Platform")
    section = st.radio(
        "Section", ["📤 Upload", "💬 Chat", "🆚 Compare", "📁 Library", "📊 Stats", "🧪 Experiments"],
        horizontal=True, key="active_section", label_visibility="collapsed",
    )
    st.divider()
    if section == "📤 Upload":
        render_upload_tab(store, model, chunk_size)
    elif section == "💬 Chat":
        render_chat_tab(store, model, prompt_template)
    elif section == "🆚 Compare":
        render_compare_tab(store, model, prompt_template)
    elif section == "📁 Library":
        render_library_tab(store, model, chunk_size)
    elif section == "📊 Stats":
        render_stats_tab(store)
    elif section == "🧪 Experiments":
        render_experiments_tab()

if __name__ == "__main__":
    main()
