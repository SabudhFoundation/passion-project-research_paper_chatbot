"""
prompt.py
Goal: Build prompt strings for the LLM.
      No LLM calls, no DB access, no final answers — only prompt construction.
"""

from typing import Optional, List



def route_query(
    question: str,
    valid_pipelines: Optional[List[str]] = None,
) -> str:
    """
    Build a classifier prompt that instructs the LLM to route
    the question to exactly one pipeline.

    Args:
        question:         The user query.
        valid_pipelines:  Allowed pipeline labels. Defaults to ["rag", "non_rag"].

    Returns:
        A prompt string (not the LLM response).
    """
    if valid_pipelines is None:
        valid_pipelines = ["rag", "non_rag"]

    pipelines_str = ", ".join(f'"{p}"' for p in valid_pipelines)

    return f"""You are a query routing assistant. \
Your only job is to classify the user question into exactly one of these pipelines: {pipelines_str}.

Routing rules:
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


def summarize_rag(question: str, context: str) -> str:
    """
    Build a prompt for answering a question using retrieved context.

    Args:
        question: The user query.
        context:  Non-empty retrieved context string.
                  Callers (main.py) must guarantee this is non-empty;
                  sending an empty-context RAG prompt is a logic error
                  that main.py now catches before calling this function.

    Returns:
        A prompt string (not the LLM response).
    """
    if not context or not context.strip():
        raise ValueError(
            "summarize_rag() requires a non-empty context string. "
            "Handle the no-chunks case in the caller before invoking this function."
        )

    return f"""You are a knowledgeable research assistant. \
Answer the user's question using ONLY the provided context.

Instructions:
- Focus on information directly relevant to the question.
- If the context is insufficient to answer fully, clearly state what is missing.
- Be concise, professional, and informative.
- Do not fabricate information not present in the context.

Context:
{context.strip()}

Question: {question}

Answer:"""


def summarize_non_rag(question: str) -> str:
    """
    Build a prompt for answering a general (non-RAG) question without retrieval.

    The caller decides whether to route here based on the query classifier.

    Returns:
        A prompt string (not the LLM response).
    """
    return f"""You are a friendly and professional AI assistant specialising in AI research.

Instructions:
- Respond in a friendly, professional, and helpful tone.
- If the user is asking about AI research papers in general, guide them to ask a more
  specific question (e.g., a particular paper, method, or result).
- If the user sends a greeting, respond politely and invite a research-related question.
- Keep answers concise.

User message: {question}

Response:"""
