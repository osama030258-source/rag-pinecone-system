"""
Pinecone Client Module
-----------------------
Responsible for:
  - Loading Pinecone API key from environment
  - Creating the index if it doesn't already exist
  - Returning a connected index handle for upsert/query operations

Index config:
  - dimension: 384 (must match embedder.py's all-MiniLM-L6-v2 output)
  - metric: cosine (per assignment requirement)
  - serverless spec: AWS us-east-1 (free-tier default region)
"""

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "rag-pdf-index"
EMBEDDING_DIMENSION = 384
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"


def get_pinecone_client() -> Pinecone:
    """Loads the API key from .env and returns a Pinecone client instance."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY not found in environment. Check your .env file."
        )
    return Pinecone(api_key=api_key)


def get_or_create_index(index_name: str = INDEX_NAME):
    """
    Returns a handle to the Pinecone index, creating it first if it
    doesn't already exist. Waits until the index is ready before returning.
    """
    pc = get_pinecone_client()

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)


if __name__ == "__main__":
    # Quick manual test — run: python src/retriever/pinecone_client.py
    print(f"Connecting to Pinecone and ensuring index '{INDEX_NAME}' exists...")
    index = get_or_create_index()
    stats = index.describe_index_stats()
    print("Connected successfully.")
    print(f"Index stats: {stats}")
