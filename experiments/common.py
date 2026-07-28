import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.chunker import chunk_pages  # noqa: E402
from core.loader import load_pages  # noqa: E402

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")

# The one real question every experiment tests against. Nothing about the *answer*,
# the *retrieved chunk*, or the *similarity score* is written in advance — those are
# computed at run time against whatever real documents you provide.
TEST_QUESTION = "Why does overlapping chunks help prevent losing context at a chunk boundary?"


def parse_question_and_paths(argv: list[str], default_question: str) -> tuple[str, list[str]]:
    """Splits `--question=...` (optional) from the remaining file path arguments."""
    question = default_question
    paths = []
    for arg in argv:
        if arg.startswith("--question="):
            question = arg[len("--question="):]
        else:
            paths.append(arg)
    return question, paths


def require_paths(argv: list[str]) -> list[Path]:
    """Validates CLI args are real, readable, supported files. No fallback corpus —
    if you don't provide real documents, the experiment refuses to run rather than
    substituting bundled sample content."""
    if not argv:
        raise SystemExit(
            "Usage: python experiments/<script>.py <file1> [file2] [file3] ...\n"
            f"Provide at least one real document ({', '.join(SUPPORTED_EXTENSIONS)})."
        )
    paths = [Path(p) for p in argv]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise SystemExit(f"Unsupported type '{p.suffix}': {p} (supported: {', '.join(SUPPORTED_EXTENSIONS)})")
    return paths


def load_corpus_by_file(paths: list[Path]) -> dict[str, list[tuple[int, str]]]:
    """{filename: [(page_number, text), ...]} for each real file you provided."""
    by_file: dict[str, list[tuple[int, str]]] = {}
    for path in paths:
        pages = load_pages(path.name, path.read_bytes())
        if pages:
            by_file[path.name] = pages
    if not by_file:
        raise ValueError("None of the provided files produced any extractable text.")
    return by_file


def load_corpus(paths: list[Path]) -> list[tuple[int, str]]:
    """Flattened (page_number, text) list, page numbers offset per file."""
    pages, offset = [], 0
    for file_pages in load_corpus_by_file(paths).values():
        pages.extend((n + offset, t) for n, t in file_pages)
        offset += len(file_pages)
    return pages


def chunk_by_file(pages_by_file: dict, size: int, overlap: int) -> dict[str, list[dict]]:
    return {name: chunk_pages(pages, size=size, overlap=overlap) for name, pages in pages_by_file.items()}


def chunk_corpus_by_file(paths: list[Path], size: int, overlap: int) -> dict[str, list[dict]]:
    return chunk_by_file(load_corpus_by_file(paths), size, overlap)


def corpus_stats(pages: list[tuple[int, str]]) -> str:
    return f"{len(pages)} page(s), {sum(len(t) for _, t in pages):,} characters"


def corpus_stats_by_file(chunks_by_file: dict, label: str = "corpus") -> str:
    lines = [f"{label}: {len(chunks_by_file)} document(s)"]
    lines += [f"  - {fn}: {len(cs)} chunk(s)" for fn, cs in chunks_by_file.items()]
    return "\n".join(lines)


def boundary_overlap_chars(chunks: list[dict]) -> int:
    """Real measurement: longest suffix of chunk[i] that's also a prefix of chunk[i+1]."""
    total = 0
    for a, b in zip(chunks, chunks[1:]):
        ta, tb = a["text"], b["text"]
        for n in range(min(len(ta), len(tb)), 0, -1):
            if ta[-n:] == tb[:n]:
                total += n
                break
    return total


def pick_relevant_and_irrelevant(chunks_by_file: dict, query: str) -> tuple[tuple[str, dict], tuple[str, dict]]:
    """Most/least keyword-relevant chunk to `query`, same technique as Store.hybrid_query."""
    terms = [t for t in query.lower().split() if len(t) > 3]
    all_chunks = [(fn, c) for fn, cs in chunks_by_file.items() for c in cs]
    if len(all_chunks) < 2:
        raise ValueError("Need at least 2 chunks across the provided documents to compare relevance.")
    kw_score = lambda c: sum(c["text"].lower().count(t) for t in terms)
    ranked = sorted(all_chunks, key=lambda pair: -kw_score(pair[1]))
    return ranked[0], ranked[-1]
