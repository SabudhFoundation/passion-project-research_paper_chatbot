"""
prompt.py
Goal: Build prompt strings for the LLM.
      No LLM calls, no DB access, no final answers — only prompt construction.
"""

from typing import Optional, List



def route_query(
    question: str,
    valid_pipelines: Optional[List[str]] = None,
    memory=None,
) -> str:
    """
    Build a classifier prompt that instructs the LLM to route
    the question to exactly one pipeline.

    Args:
        question:         The user query.
        valid_pipelines:  Allowed pipeline labels. Defaults to ["rag", "non_rag"].
        memory:           Optional list of recent interactions (dicts with 'question', 'answer', 'route').

    Returns:
        A prompt string (not the LLM response).
    """
    if valid_pipelines is None:
        valid_pipelines = ["rag", "non_rag"]

    pipelines_str = ", ".join(f'"{p}"' for p in valid_pipelines)

    memory_block = ""
    if memory:
        formatted = "\n\n".join(
            f"Q: {m['question']}\nRoute: {m['route']}"
            for m in memory
        )
        memory_block = f"""Recent conversation history (context for routing):
{formatted}

Consider this history when routing the current question. If this is a follow-up, maintain consistency with previous routing decisions.

"""

    return f"""You are a query routing assistant. \
Your only job is to classify the user question into exactly one of these pipelines: {pipelines_str}.

{memory_block}Routing rules:
- Use "rag" for: questions about AI research papers, specific models, methods, \
experiments, citations, technical retrieval needs, or anything requiring knowledge \
from a document store.
- Use "non_rag" for: general conversation, greetings, or vague questions that \
are clearly not about research or technology.

CRITICAL RULE: If the user asks for a definition, explanation, or summary \
of a technical concept (e.g., 'What is X?', 'Explain Y'), you MUST route to "rag". \
Do not rely on your general knowledge for technical terms; always route these to the documents.

You MUST respond with ONLY valid JSON — no preamble, no explanation, no markdown fences.
Your entire response must be exactly one of these two tokens:
{{"route": "rag"}}
{{"route": "non_rag"}}

User question: {question}"""


def summarize_rag(question: str, context: str, memory=None) -> str:
    """
    Build a prompt for answering a question using retrieved context.
    Optionally considers recent conversation memory for relevance.

    Args:
        question: The user query.
        context:  Non-empty retrieved context string.
        memory:   Optional list of recent interactions (dicts with 'question', 'answer', 'route').

    Returns:
        A prompt string (not the LLM response).
    """
    if not context or not context.strip():
        raise ValueError(
            "summarize_rag() requires a non-empty context string. "
            "Handle the no-chunks case in the caller before invoking this function."
        )

    memory_block = ""

    if memory:
        formatted = "\n\n".join(
            f"Q: {m['question']}\nA: {m['answer']}"
            for m in memory
        )

        memory_block = f"""
Previous conversation (may or may not be relevant):
{formatted}

IMPORTANT:
- First, decide if the previous conversation is relevant to the current question.
- If relevant, use it to improve the answer.
- If NOT relevant, completely ignore it.
"""

    return f"""You are a knowledgeable research assistant.

{memory_block}

Instructions:
- Use the retrieved context as the primary source of truth.
- You may use relevant previous conversation ONLY to improve continuity or clarity.
- If context is insufficient, say so clearly.
- Be concise and accurate.
- Do not hallucinate.

Context:
{context.strip()}

Question: {question}

Answer:"""


def summarize_non_rag(question: str, memory=None) -> str:
    """
    Build a prompt for answering a general (non-RAG) question without retrieval.
    Optionally considers recent conversation memory for continuity.

    Args:
        question: The user query.
        memory:   Optional list of recent interactions (dicts with 'question', 'answer', 'route').

    Returns:
        A prompt string (not the LLM response).
    """
    memory_block = ""

    if memory:
        formatted = "\n\n".join(
            f"Q: {m['question']}\nA: {m['answer']}"
            for m in memory
        )

        memory_block = f"""
Previous conversation:
{formatted}

IMPORTANT:
- Use previous conversation ONLY if relevant.
- Otherwise ignore it.
"""

    return f"""You are a friendly and professional AI assistant.

{memory_block}

User message: {question}

Response:"""
