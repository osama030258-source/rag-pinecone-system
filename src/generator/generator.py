"""
Generation Module
--------------------
Responsible for:
  - Building a strict, context-only prompt from retrieved chunks
  - Calling the Groq LLM API to generate an answer
  - Enforcing the hallucination-prevention fallback message when
    no relevant context was retrieved

This module does NOT decide relevance itself — that's the retriever's job
(via similarity_threshold). If retriever returns an empty list, this
module short-circuits and returns the fallback message without even
calling the LLM (saves an API call and guarantees no hallucination).
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

FALLBACK_MESSAGE = "The answer is not available in the provided document."

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a document question-answering assistant.

STRICT RULES:
1. Answer ONLY using the provided context below. Do not use any outside knowledge.
2. If the context does not contain enough information to answer the question, respond with exactly:
   "The answer is not available in the provided document."
3. Do not guess, infer beyond the context, or fabricate details.
4. Keep answers concise and directly grounded in the context.
5. When helpful, briefly mention which part of the context supports your answer.
"""


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in environment. Check your .env file.")
    return Groq(api_key=api_key)


def build_context_block(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a labeled context block for the prompt,
    so the LLM (and we, when debugging) can see which page each piece came from.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {i} | Page {chunk['page_number']} | Score {chunk['score']:.3f}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generates an answer strictly from retrieved_chunks.

    Args:
        query: the user's question
        retrieved_chunks: output of retriever.retrieve_relevant_chunks()

    Returns:
        {
            "answer": "...",
            "sources": [ {page_number, text, score, document_name}, ... ],
            "grounded": True/False   # False if fallback was triggered
        }
    """
    if not retrieved_chunks:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "grounded": False,
        }

    context_block = build_context_block(retrieved_chunks)

    user_prompt = f"""Context:
{context_block}

Question: {query}

Answer strictly using the context above."""

    client = _get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temperature to reduce creative drift/hallucination
        max_tokens=600,
    )

    answer_text = response.choices[0].message.content.strip()

    is_fallback = FALLBACK_MESSAGE.lower() in answer_text.lower()

    return {
        "answer": answer_text,
        "sources": [] if is_fallback else retrieved_chunks,
        "grounded": not is_fallback,
    }


if __name__ == "__main__":
    # Quick manual test — run: python -m src.generator.generator "<namespace>" "<query>"
    import sys
    from src.retriever.retriever import retrieve_relevant_chunks

    if len(sys.argv) < 3:
        print('Usage: python -m src.generator.generator "<namespace>" "<query>"')
        sys.exit(1)

    namespace_arg = sys.argv[1]
    query_arg = sys.argv[2]

    chunks = retrieve_relevant_chunks(query_arg, namespace_arg)
    result = generate_answer(query_arg, chunks)

    print(f"Question: {query_arg}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Grounded: {result['grounded']}")
    print(f"Sources used: {len(result['sources'])}")
    for s in result["sources"]:
        print(f"  - Page {s['page_number']} (score {s['score']:.3f})")
