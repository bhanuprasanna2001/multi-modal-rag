"""RAG pipeline: hybrid retrieval + OpenAI answer generation with cost/latency tracking."""

from __future__ import annotations

import os
from typing import List, Dict, Any

from openai import OpenAI

from .index import search
from .utils import Metrics, Timer

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about electronics datasheets. "
    "Use ONLY the provided context to answer. If the context doesn't contain the "
    "answer, say 'I don't have enough information to answer this question.'\n\n"
    "Rules:\n"
    "- Cite every factual claim with (DOC_ID:PAGE) using the tags from the context\n"
    "- Be concise and specific\n"
    "- If multiple documents contain relevant info, cite all of them\n"
    "- For numerical values, always include units"
)


def _format_context(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved hits into a numbered context block for the LLM."""
    lines: List[str] = []
    for i, hit in enumerate(hits, 1):
        tag = f"[{hit.get('doc_id')}:{hit.get('page')}]"
        kind = "Image caption" if hit["type"] == "caption" else "Text"
        lines.append(f"{i}. {kind} {tag}:\n{hit['content']}")
    return "\n\n".join(lines)


def answer_question(
    question: str,
    k: int = 5,
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """Retrieve relevant context and generate an answer using OpenAI.

    Returns answer, source hits, and cost/latency metrics.
    Raises EnvironmentError if OPENAI_API_KEY is not set.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is required. Set it in your environment:\n"
            "  export OPENAI_API_KEY='your-key-here'"
        )

    metrics = Metrics()
    total_timer = Timer()

    with total_timer:
        # Retrieve (hybrid: dense + BM25)
        with Timer() as retrieval_timer:
            hits = search(question, k=k, artifacts_dir=artifacts_dir, metrics=metrics)
        metrics.retrieval_ms = retrieval_timer.ms

        context = _format_context(hits)

        # Generate answer
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI()

        with Timer() as gen_timer:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
                ],
                temperature=0.0,
                max_tokens=500,
            )
        metrics.generation_ms = gen_timer.ms

        # Track generation cost
        usage = response.usage
        if usage:
            metrics.add_generation_cost(model, usage.prompt_tokens, usage.completion_tokens)

        answer = response.choices[0].message.content.strip()

    metrics.total_ms = total_timer.ms

    return {
        "answer": answer,
        "hits": hits,
        "context": context,
        "metrics": metrics.to_dict(),
    }
