from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[dict]:
    """pages: [(page_number, text), ...] -> [{"text", "page"}, ...]"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = []
    for page_num, text in pages:
        text = " ".join(text.split())  # normalize whitespace
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append({"text": piece.strip(), "page": page_num})
    return chunks
