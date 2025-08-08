from __future__ import annotations
import os
from typing import List, Dict, Any
from loguru import logger
from .index import search
from .utils import Paths

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about electronics datasheets. "
    "Use only the provided context snippets. If the answer is not contained in the context, say 'I don't know'. "
    "Cite evidence using (DOC:PAGE) after any factual statements. Keep answers concise."
)


def _format_context(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for h in hits:
        tag = f"({h.get('doc_id')}:{h.get('page')})"
        if h["type"] == "caption":
            lines.append(f"Image caption {tag}: {h['content']}")
        else:
            # Truncate long text
            content = h["content"]
            if len(content) > 600:
                content = content[:600] + " …"
            lines.append(f"Text {tag}: {content}")
    return "\n".join(lines)


def _local_compose_answer(question: str, context: str) -> str:
    # Minimal fallback when no API key: rule-based extract or conservative response
    # Try to echo most relevant sentence containing key terms; otherwise say I don't know
    import re
    q_terms = [w for w in re.findall(r"[A-Za-z0-9_+-]+", question.lower()) if len(w) > 2]
    best_line = None
    best_score = 0
    candidate_lines = context.splitlines()
    # Extra boost for spec / temperature patterns
    import re
    temp_q = bool(re.search(r"temp|°|deg|range", question.lower()))
    temp_pat = re.compile(r"-?\d+\s*[°º]?[cf]\s*(?:to|-|–)\s*-?\d+\s*[°º]?[cf]", re.IGNORECASE)
    for line in candidate_lines:
        lc = line.lower()
        score = sum(1 for t in q_terms if t in lc)
        if temp_q and temp_pat.search(lc):
            score += 5
        if "spec" in lc:
            score += 1
        if score > best_score:
            best_score = score
            best_line = line
    if best_line and best_score >= 2:
        return best_line
    return "I don't know. (no supporting context)"


def answer_question(question: str, k: int = 6, artifacts_dir: str = "artifacts", use_openai: bool | None = None) -> Dict[str, Any]:
    hits = search(question, k=k, artifacts_dir=artifacts_dir)
    context = _format_context(hits)

    client = None
    if use_openai is None:
        use_openai = bool(os.getenv("OPENAI_API_KEY"))
    if use_openai and OpenAI is not None:
        try:
            client = OpenAI()
        except Exception as e:
            logger.warning(f"OpenAI client init failed: {e}")
            client = None

    if client:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ]
        try:
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            ans = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")
            ans = _local_compose_answer(question, context)
    else:
        ans = _local_compose_answer(question, context)

    # Ensure at least one (DOC:PAGE) citation if we have hits
    has_citation = "(" in ans and ")" in ans and ":" in ans
    if not has_citation and hits:
        tag = f"({hits[0].get('doc_id')}:{hits[0].get('page')})"
        ans = f"{ans} {tag}".strip()

    return {
        "answer": ans,
        "hits": hits,
        "context": context,
    }


if __name__ == "__main__":
    out = answer_question("What is the operating temperature range?")
    print(out["answer"]) 
