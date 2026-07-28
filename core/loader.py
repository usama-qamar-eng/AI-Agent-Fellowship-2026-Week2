import io
from docx import Document
from pypdf import PdfReader

def load_pages(filename: str, file_bytes: bytes) -> list[tuple[int, str]]:
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        return [(n, t) for n, t in pages if t.strip()]

    if ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)
        return [(1, text)] if text.strip() else []

    # txt / md — no real page concept, treat as one page
    text = file_bytes.decode("utf-8", errors="ignore")
    return [(1, text)] if text.strip() else []
