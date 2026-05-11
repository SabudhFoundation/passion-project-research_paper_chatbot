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
        memory:           Optional list of recent interactions
                          (dicts with 'question', 'answer', 'route').

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

        memory_block = f"""Recent conversation history:
{formatted}

FOLLOW-UP DETECTION & ROUTING:
- If the current question references concepts, keywords, or pronouns from the 
    previous conversation (e.g., "it", "that", "the method", etc.), it is a FOLLOW-UP.
- For follow-ups to technical discussions (previous route was "rag"):
    Usually route to "rag" unless the current message is clearly a greeting, thanks, or unrelated conversational message.
- For follow-ups to non-technical discussions (previous route was "non_rag"):
    Evaluate the current question independently using the routing rules below.
- If the current question is NOT a follow-up, ignore previous routes and 
    classify based solely on the content of the new question.

"""

    return f"""You are a query routing assistant.

Your ONLY job is to classify the user question into exactly one pipeline:
{pipelines_str}

ROUTING RULES:

Use "rag" for:
- AI research papers
- machine learning concepts
- deep learning
- NLP
- retrieval systems
- technical concepts
- model architectures
- algorithms
- experiments
- research methodologies
- technical explanations
- summaries
- definitions
- citations
- anything related to AI or technical research

Use "non_rag" ONLY for:
- greetings
- thanks
- conversational pleasantries
- questions clearly unrelated to AI research papers or technical concepts

CRITICAL RULES:
- If the user asks for a definition, explanation, or summary of ANY
  technical concept, ALWAYS route to "rag".
- Never reject technical or AI-related questions.
- Any question related to AI, ML, NLP, LLMs, transformers, RAG,
  vector databases, embeddings, agents, or research papers MUST go to "rag".
- Questions unrelated to research or technical AI topics MUST go to "non_rag".

{memory_block}

You MUST respond with ONLY valid JSON.
Do not include explanations, markdown, or extra text.

Valid responses:
{{"route": "rag"}}
{{"route": "non_rag"}}

User question: {question}
"""


def summarize_rag(question: str, context: str, memory=None) -> str:
    """
    Build a prompt for answering a question using retrieved context.
    Optionally considers recent conversation memory for relevance.

    Args:
        question: The user query.
        context:  Non-empty retrieved context string.
        memory:   Optional list of recent interactions
                   (dicts with 'question', 'answer', 'route').

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
- First decide whether the previous conversation is relevant.
- If relevant, use it to improve continuity and clarity.
- If NOT relevant, completely ignore it.
"""

    return f"""You are a focused AI research-paper assistant.

{memory_block}

INSTRUCTIONS:
- Use the retrieved context as the PRIMARY source of truth.
- Ground your answer in the provided research-paper context.
- Be accurate, concise, and technical when needed.
- Do NOT hallucinate facts not supported by context.
- If the retrieved context partially answers the question,
    clearly mention the limitation.
- If the question asks for a definition or explanation:
    - first rely on retrieved research context
    - if context is incomplete, you may use general technical knowledge
    - clearly indicate when information is not fully grounded in retrieved papers
- Never invent citations, experiments, or results.
- Keep answers focused on AI and technical research topics.

Retrieved context:
{context.strip()}

Question:
{question}

Answer:
"""


def summarize_non_rag(question: str, memory=None) -> str:
    """
    Build a prompt for handling greetings and rejecting unrelated questions.

    Args:
        question: The user query.
        memory:   Optional list of recent interactions
                   (dicts with 'question', 'answer', 'route').

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

    return f"""You are a research paper chatbot focused ONLY on AI and technical research papers.

{memory_block}

BEHAVIOR RULES:

1. If the user message is a greeting:
- respond politely
- keep the response short
- guide the user toward research-related questions

Example:
"Hello! I am a research paper chatbot focused on AI and technical research papers. Ask me about models, methods, experiments, or technical concepts."

2. If the user question is unrelated to AI research papers or technical concepts:
- politely refuse to answer
- clearly state that you are a research paper chatbot
- ask the user to ask research-related questions instead

3. NEVER behave like a general-purpose assistant.

4. Do not answer unrelated informational or task-oriented questions.
    Simple greetings and conversational pleasantries are allowed.

5. Keep responses concise and professional.

User message:
{question}

Response:
"""
