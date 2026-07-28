from config import DEFAULT_PROMPT_TEMPLATE, PROMPT_TEMPLATES, TOP_K
from core.llm import ask
from core.store import Store

def answer_question(
    store: Store, model: str, question: str, chat_history: list[dict],
    doc_filter: str | None = None, use_hybrid: bool = False,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> tuple[str, list[dict], int]:
    retrieve = store.hybrid_query if use_hybrid else store.query
    chunks = retrieve(question, top_k=TOP_K, doc_filter=doc_filter)

    if not chunks:
        return "I could not find this in the uploaded documents.", [], 0

    context = "\n\n".join(
        f"[{c['source']}, page {c['page']}, chunk #{c['chunk_id']}]\n{c['text']}" for c in chunks
    )
    system_prompt = PROMPT_TEMPLATES[prompt_template].format(context=context)

    # keep last 6 turns for follow-up context, then the new question
    history = chat_history[-6:] + [{"role": "user", "content": question}]
    answer, tokens = ask(model, system_prompt, history)
    return answer, chunks, tokens

def compare_documents(
    store: Store, model: str, doc_a: str, doc_b: str, question: str = "",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> tuple[str, list[dict], int]:
    """Retrieve top chunks from each document separately, then ask the LLM to compare them."""
    q = question.strip() or f"Compare and contrast '{doc_a}' and '{doc_b}'. Highlight key similarities and differences."
    chunks_a = store.query(q, top_k=TOP_K, doc_filter=doc_a)
    chunks_b = store.query(q, top_k=TOP_K, doc_filter=doc_b)
    all_chunks = chunks_a + chunks_b

    if not all_chunks:
        return "I could not find enough content in these documents to compare.", [], 0

    context = "\n\n".join(
        f"[{c['source']}, page {c['page']}, chunk #{c['chunk_id']}]\n{c['text']}" for c in all_chunks
    )
    system_prompt = PROMPT_TEMPLATES[prompt_template].format(context=context)
    answer, tokens = ask(model, system_prompt, [{"role": "user", "content": q}])
    return answer, all_chunks, tokens
