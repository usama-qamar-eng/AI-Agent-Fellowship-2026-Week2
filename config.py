import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (OpenRouter — OpenAI-compatible endpoint) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODELS = {
    "Nemotron 3 Ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "Gemma 4 31B": "google/gemma-4-31b-it:free",
}
 
LLM_TEMPERATURE = 0.2

# --- Chunking (selectable in the sidebar; see experiments/exp1_chunk_size.py and
#     exp2_chunk_overlap.py for why 500/50 is the default) ---
CHUNK_SIZE_OPTIONS = [300, 500, 1000]
DEFAULT_CHUNK_SIZE = 500
CHUNK_OVERLAP = 50     

# --- Embeddings + Vector store ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
COLLECTION_NAME = "documents"

# --- Retrieval ---
TOP_K = 4

# --- Upload ---
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")

# --- Prompt templates (selectable in the sidebar; see experiments/exp3_prompt_templates.py
#     for the comparison behind these three) ---
PROMPT_TEMPLATES = {
    "Strict grounding (default)": """You are an enterprise document assistant. Answer ONLY using the \
context passages below — each is tagged with its source document and page.

Rules:
1. If the answer isn't in the context, say "I could not find this in the uploaded \
documents." Never guess or use outside knowledge.
2. Cite every claim as [document, page X, chunk #Y].
3. Be concise and stay grounded in the retrieved text.

Context:
{context}""",
    "Concise": """Answer the question in as few sentences as possible, using only the \
context below. Cite the source briefly.

Context:
{context}""",
    "Verbose reasoning": """You are a thorough enterprise assistant. Using only the context \
below, explain your reasoning step by step before giving a final answer. Cite every claim \
as [document, page X, chunk #Y]. If the answer isn't in the context, say so explicitly.

Context:
{context}""",
}
DEFAULT_PROMPT_TEMPLATE = "Strict grounding (default)"
