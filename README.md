# Document Intelligence Platform

A minimal, production shaped RAG app: upload PDFs/DOCX/TXT/MD, ask questions, get grounded
answers with citations.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your OpenRouter key (https://openrouter.ai/keys)
                             # optionally set APP_PASSWORD too, to require a login screen
streamlit run app.py
```

## Sidebar controls
Three real, functional selectors not static text:
- **LLM Model** switches between OpenRouter's Nemotron 3 Ultra/Gemma 4 31B
- **Chunk size** 300/500/1000 chars
- **Prompt template** Strict grounding (default) / Concise / Verbose reasoning, applied
  to every chat and comparison answer immediately, no restart needed


Navigation between Upload/Chat/Compare/Library/Stats uses `st.radio`, not `st.tabs`
`st.tabs` executes every tab's code on every rerun even when hidden, which risked widget
state bleeding between sections. `st.radio` + an if/elif chain guarantees only the visible
section's code runs at all.

## Architecture
```
User -> Streamlit UI -> core/loader.py   (PDF/DOCX/TXT/MD -> pages)
                      -> core/chunker.py (pages -> overlapping chunks, via LangChain's RecursiveCharacterTextSplitter)
                      -> core/store.py   (ChromaDB + sentence-transformers embeddings)
                      -> core/pipeline.py (retrieve top-k chunks -> build grounded prompt)
                      -> core/llm.py     (OpenRouter chat completion)
                      -> answer + citations back to UI
```

## Bonus features implemented
- **DOCX support** `python-docx` extracts paragraph text (no reliable page numbers in
  Word, so treated as one page same as TXT/MD) 
- **Metadata filtering** scope chat/search to one document via dropdown.
- **Hybrid search** toggle blends semantic similarity with keyword overlap (`Store.hybrid_query`).
- **Document comparison** Compare tab retrieves from two docs separately, asks the LLM to synthesize.
- **Auto-summarization** one LLM call per upload produces a 2-3 sentence summary.
- **Suggested questions** same call also produces 3 example questions, clickable to prefill chat.
- **Chat export** download the full transcript as markdown.
- **Token usage dashboard** real `total_tokens` from OpenRouter tracked per-model.
- **Dark mode** provided natively by Streamlit (⋮ menu → Settings → Theme), not
  reimplemented with custom CSS.


## Why these choices
- **LangChain for chunking only** `RecursiveCharacterTextSplitter` splits on paragraph/
  sentence/word boundaries where possible before falling back to a hard cut, which is a
  real quality improvement over a naive fixed-size slice. Scoped to one file
  (`core/chunker.py`) rather than pulled through the rest of the app, so the dependency
  stays contained and doesn't dictate the shape of the other modules.
- **ChromaDB** embeds and stores chunks with metadata (`source`, `page`) in one call;
  no separate embedding module needed.
- **OpenRouter over a direct SDK** `requests` + one function (`core/llm.py`) is enough
  for a single chat completions call; no SDK dependency required.
- **File-based persistence** (`data/chroma_db/`) — survives restarts, no external DB to run.

## Known limitations
- Auth is a single shared password (`APP_PASSWORD` in `.env`), not per-user accounts
  there's still no concept of separate users, just one gate in front of the whole app.
  Matches the spec's "basic login," not a full multi-tenant auth system.
- DOCX pages are approximate Word has no fixed page concept, so a DOCX upload is
  treated as one page, same as TXT/MD. Page citations for DOCX sources will always say
  "page 1"; PDF citations remain accurate to the real page.
- "Refresh" needs the original file bytes cached in the current session
  (`st.session_state.raw_files`); after a server restart, re-upload instead.
- Summarization/suggested-questions run once per upload and aren't regenerated on refresh
  unless you re-run the upload flow.
- Hybrid search re-ranks an over fetched candidate set with keyword overlap a real BM25
  index would be more principled at large document counts, but adds a dependency for
  marginal gain at this scale.


## Demo Video Link

https://drive.google.com/file/d/1U0kvT5UYia_AmdJiW9GB8RyzkeNh3jNq/view?usp=drive_link