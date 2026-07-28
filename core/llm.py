import requests
from config import LLM_TEMPERATURE, OPENROUTER_API_KEY, OPENROUTER_URL

def ask(model: str, system_prompt: str, history: list[dict]) -> tuple[str, int]:
    if not OPENROUTER_API_KEY:
        raise ValueError("No OpenRouter API key set (check your .env file).")

    messages = [{"role": "system", "content": system_prompt}] + history
    r = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": LLM_TEMPERATURE},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:300]}")

    data = r.json()
    content = data["choices"][0]["message"]["content"] or ""
    tokens = data.get("usage", {}).get("total_tokens", max(1, len(content) // 4))
    return content, tokens

def summarize_and_suggest(model: str, doc_name: str, text: str) -> tuple[str, list[str], int]:
    """One call, two jobs: a short summary + 3 example questions the user could ask.
    Combining them into a single request halves the token cost vs. two separate calls."""
    prompt = (
        f"Document: {doc_name}\n\n{text[:6000]}\n\n"
        "Respond in exactly this format:\n"
        "SUMMARY: <2-3 sentence summary>\n"
        "QUESTIONS:\n1. <question>\n2. <question>\n3. <question>"
    )
    content, tokens = ask(model, "You summarize documents and write example questions about them.",
                          [{"role": "user", "content": prompt}])

    summary, questions = content.strip(), []
    if "QUESTIONS:" in content:
        summary_part, q_part = content.split("QUESTIONS:", 1)
        summary = summary_part.replace("SUMMARY:", "").strip()
        for line in q_part.strip().splitlines():
            line = line.strip().lstrip("0123456789.").strip()
            if line:
                questions.append(line)
    return summary, questions[:3], tokens
