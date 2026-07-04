"""
Retrieval Module
------------------
Responsible for:
  - Embedding the user's query using the same model as the chunks
  - Querying Pinecone for the top-k most similar chunks (cosine similarity)
  - Filtering results by a minimum similarity threshold
  - Optionally filtering by metadata (e.g. specific page number)

Returns retrieved chunks with their similarity score and full metadata,
so the generator module can build a context-grounded prompt and the UI
can display source attribution (page number, excerpt, score).
"""

from src.retriever.pinecone_client import get_or_create_index
from src.embedder.embedder import embed_texts

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3  # cosine similarity, range -1 to 1 (typically 0 to 1 for text)


def retrieve_relevant_chunks(
    query: str,
    namespace: str,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    page_filter: int = None,
) -> list[dict]:
    """
    Retrieves the top-k relevant chunks for a query from a given namespace.

    Args:
        query: the user's natural language question
        namespace: which document's namespace to search (from upsert_chunks)
        top_k: number of chunks to retrieve
        similarity_threshold: minimum cosine similarity score to keep a result
        page_filter: if provided, restrict search to a specific page number

    Returns:
        List of dicts:
            [
                {
                    "score": 0.842,
                    "text": "chunk text...",
                    "page_number": 4,
                    "document_name": "sample.pdf",
                    "chunk_id": "uuid"
                },
                ...
            ]
        Empty list if nothing meets the threshold (caller should then
        return the "not available in document" fallback message).
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    index = get_or_create_index()

    query_vector = embed_texts([query])[0]

    filter_dict = None
    if page_filter is not None:
        filter_dict = {"page_number": {"$eq": page_filter}}

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filter_dict,
    )

    matches = []
    for match in results.get("matches", []):
        if match["score"] >= similarity_threshold:
            matches.append({
                "score": match["score"],
                "text": match["metadata"]["text"],
                "page_number": match["metadata"]["page_number"],
                "document_name": match["metadata"]["document_name"],
                "chunk_id": match["metadata"]["chunk_id"],
            })

    return matches


if __name__ == "__main__":
    # Quick manual test — run: python -m src.retriever.retriever "<namespace>" "<query>"
    import sys

    if len(sys.argv) < 3:
        print('Usage: python -m src.retriever.retriever "<namespace>" "<query>"')
        sys.exit(1)

    namespace_arg = sys.argv[1]
    query_arg = sys.argv[2]

    results = retrieve_relevant_chunks(query_arg, namespace_arg)

    print(f"Query: {query_arg}")
    print(f"Namespace: {namespace_arg}")
    print(f"Matches found: {len(results)}\n")

    for i, r in enumerate(results, start=1):
        print(f"--- Match {i} (score: {r['score']:.3f}, page {r['page_number']}) ---")
        print(r["text"][:200])
        print()
