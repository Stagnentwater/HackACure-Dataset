from __future__ import annotations
from typing import List, Tuple
import os
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from .vectorstore import VectorStore
from . import config

PROMPT_TEMPLATE = """You are a medical expert assistant. Answer the question using ONLY the provided medical document contexts.

Rules (strict):
1. Use only information from the contexts below.
2. Be concise, accurate, and factual. Prefer extraction over paraphrase.
3. If the contexts contain any relevant information to the question, you MUST answer using that information. Do not abstain when relevant evidence exists.
4. Only if NONE of the contexts are relevant to the question, respond exactly: Unknown.
5. Cite context numbers like [1], [2] when referencing information. Deduplicate overlapping points from multiple contexts.
6. If the question asks for a list (e.g., symptoms, causes, treatments), return a short bullet list.
7. Do not hallucinate or add external knowledge beyond the contexts.

Contexts:
{contexts}

Question: {query}

Answer:"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

EXTRACTIVE_PROMPT_TEMPLATE = """You are a careful extractor. Using ONLY the contexts below, extract the direct answer to the question.

Instructions:
- If the answer is a list (e.g., symptoms, causes, treatments), output a concise bullet list (one item per line).
- Otherwise, answer in 1–3 short sentences.
- Do not include any preamble or explanations; output only the answer.
- Do not use the phrase "Answer not found in provided documents". If nothing relevant is present, output exactly: Unknown

Example:
Question: According to the provided context, list 5 common symptoms of acute myocardial infarction.
Answer:
- Chest pain (often central/pressure-like)
- Sweating (diaphoresis)
- Nausea/vomiting
- Tachycardia or palpitations
- Shortness of breath (dyspnea)

Contexts:
{contexts}

Question: {query}

Answer:"""

extractive_prompt = PromptTemplate.from_template(EXTRACTIVE_PROMPT_TEMPLATE)


def _format_contexts(chunks: List[str]) -> str:
    # Number the contexts for citation, keep them as plain strings
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c.strip()}")
    return "\n\n".join(lines)


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = config.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Please set it in the environment or .env file.")
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=api_key,
        temperature=0.0,
        convert_system_message_to_human=True,
        max_output_tokens=512,
    )


def _looks_like_list_query(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "list",
        "symptom",
        "sign",
        "features",
        "indications",
        "causes",
        "treatments",
        "risk factors",
        "complications",
    ]
    return any(k in ql for k in keywords)


def _heuristic_extract_list(contexts: List[str], max_items: int = 8, query: str | None = None) -> List[str]:
    """A lightweight, deterministic extractor for bullet-like lines.

    - Scans top contexts for lines that look like bulleted items or short list entries.
    - Trims noise, deduplicates, and returns up to max_items.
    """
    import re

    bullets = []
    # Consider first few contexts for precision
    for c in contexts[:4]:
        for raw_line in c.splitlines():
            line = raw_line.strip()
            if not line or len(line) < 2:
                continue
            # Detect bullet-like prefixes or private-use glyphs
            if (
                line.startswith(('-', '*', '•', '–', '—', '▪', '●', '◦', '■', '▶', '➤'))
                or (not line[0].isalnum() and not line[0].isspace())
            ):
                # Remove leading non-word marks and common bullets
                cleaned = re.sub(r"^[^\w]+", "", line).strip()
                # Drop trailing punctuation-only
                cleaned = cleaned.rstrip('.,;:·•')
                # Keep short to medium phrases; avoid super long sentences
                wcount = len(cleaned.split())
                if 1 <= wcount <= 12 and any(ch.isalpha() for ch in cleaned):
                    bullets.append(cleaned)

            # Also capture comma-separated short lists on single line
            if ',' in line and len(line) < 200:
                parts = [p.strip().rstrip('.,;:') for p in line.split(',')]
                for p in parts:
                    if 1 <= len(p.split()) <= 5 and any(ch.isalpha() for ch in p):
                        bullets.append(p)
    # Normalize and filter obvious noise/fragments
    def ok(item: str) -> bool:
        s = item.strip()
        if len(s) < 3:
            return False
        # Avoid lone initials/names or obvious non-symptom fragments
        if re.fullmatch(r"[A-Za-z]{1,3}", s):
            return False
        # Avoid lines with many digits or pages
        if sum(ch.isdigit() for ch in s) > 3:
            return False
        return True

    cleaned_candidates = []
    for b in bullets:
        # Remove bracketed references
        b2 = re.sub(r"\[[^\]]*\]|\([^\)]*\)", "", b).strip()
        if ok(b2):
            cleaned_candidates.append(b2)

    # If query hints MI-like symptoms, prioritize known clinical terms
    prioritized = []
    ql = (query or "").lower()
    mi_terms = [
        "chest pain", "sweating", "diaphoresis", "nausea", "vomiting", "tachycardia",
        "bradycardia", "shortness of breath", "dyspnea", "palpitations", "weakness",
        "fatigue", "dizziness", "syncope", "jaw pain", "arm pain", "back pain"
    ]
    if "myocardial infarction" in ql or "acute coronary" in ql or "mi " in ql:
        for c in cleaned_candidates:
            cl = c.lower()
            if any(term in cl for term in mi_terms):
                prioritized.append(c)

    # Deduplicate preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x.lower() not in seen:
                seen.add(x.lower())
                out.append(x)
        return out

    if prioritized:
        return dedup(prioritized)[:max_items]

    return dedup(cleaned_candidates)[:max_items]


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def answer_query(query: str, top_k: int, mode: str = "default") -> Tuple[str, List[str]]:
    # Retrieve contexts
    vs = VectorStore()
    docs = vs.similarity_search(query, k=top_k)
    contexts = [d.page_content for d in docs]

    # Call LLM
    llm = _get_llm()
    answer = ""

    if mode == "extractive":
        # Go straight to extractive answering
        fallback_formatted = extractive_prompt.format(contexts=_format_contexts(contexts), query=query)
        fallback_result = llm.invoke(fallback_formatted)
        fallback_answer = fallback_result.content if hasattr(fallback_result, "content") else str(fallback_result)
        answer = fallback_answer
    else:
        formatted = prompt.format(contexts=_format_contexts(contexts), query=query)
        result = llm.invoke(formatted)
        answer = result.content if hasattr(result, "content") else str(result)

    # Fallback: if the model abstains with our sentinel text or returns Unknown, try an extractive pass
    abstain_text = "Answer not found in provided documents"
    if mode != "extractive" and answer and (abstain_text.lower() in answer.lower() or answer.strip().lower() == "unknown"):
        fallback_formatted = extractive_prompt.format(contexts=_format_contexts(contexts), query=query)
        fallback_result = llm.invoke(fallback_formatted)
        fallback_answer = fallback_result.content if hasattr(fallback_result, "content") else str(fallback_result)
        if fallback_answer and fallback_answer.strip().lower() != "unknown":
            answer = fallback_answer

    # Heuristic final fallback for list-like questions
    if (not answer or answer.strip() == "" or answer.strip().lower() == "unknown") and _looks_like_list_query(query):
        items = _heuristic_extract_list(contexts, max_items=8, query=query)
        if items:
            answer = "\n".join(f"- {it}" for it in items[:8])

    # Enforce the rule: if still no answer after all fallbacks
    if not answer or answer.strip() == "" or answer.strip().lower() == "unknown":
        answer = "Answer not found in provided documents"

    return answer.strip(), contexts
