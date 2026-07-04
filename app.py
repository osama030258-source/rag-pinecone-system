"""
RAG System - Streamlit App
----------------------------
Main entrypoint. Ties together: loader -> chunker -> embedder -> Pinecone
upsert -> retriever -> generator, with a UI for upload, querying, and
source attribution.

Mandatory enhancements implemented here:
  - Multi-document support (namespace per document, selectable in sidebar)
  - Query history (session memory)
  - Adjustable chunk size (slider, applied at processing time)
  - Adjustable top-k retrieval (slider)
  - Metadata filtering (optional page number filter)
  - Confidence scoring display (similarity score shown per source)
  - Logging user queries (to a local log file)
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.loader.pdf_loader import load_pdf, PDFLoadError
from src.chunker.chunker import chunk_pages
from src.embedder.embedder import embed_chunks
from src.retriever.pinecone_client import get_or_create_index
from src.retriever.upsert import upsert_chunks, sanitize_namespace
from src.retriever.retriever import retrieve_relevant_chunks
from src.generator.generator import generate_answer

# ---------- Logging setup (mandatory enhancement: query logging) ----------
logging.basicConfig(
    filename="query_log.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="RAG PDF Q&A System", layout="wide")

# ---------- Session state initialization ----------
if "documents" not in st.session_state:
    st.session_state.documents = {}  # {namespace: {"display_name": ..., "chunk_count": ...}}
if "query_history" not in st.session_state:
    st.session_state.query_history = []  # list of {question, answer, namespace, timestamp}

st.title("📄 RAG System — PDF Question Answering (Pinecone)")
st.caption("Upload a PDF, ask questions, get answers grounded strictly in the document.")

# ============================================================
# SIDEBAR — Upload & Settings
# ============================================================
with st.sidebar:
    st.header("1. Upload & Process")

    uploaded_file = st.file_uploader("Upload a PDF (max 20MB)", type=["pdf"])

    chunk_size = st.slider("Chunk size (characters)", min_value=300, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=500, value=150, step=50)

    process_clicked = st.button("🔄 Process PDF", type="primary", disabled=uploaded_file is None)

    if process_clicked and uploaded_file is not None:
        try:
            if chunk_overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than chunk size.")
            else:
                save_path = DATA_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Extracting text..."):
                    pages = load_pdf(str(save_path))

                with st.spinner(f"Chunking ({len(pages)} pages)..."):
                    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
                    embedded = embed_chunks(chunks)

                with st.spinner("Upserting into Pinecone..."):
                    result = upsert_chunks(embedded)

                st.session_state.documents[result["namespace"]] = {
                    "display_name": uploaded_file.name,
                    "chunk_count": result["upserted_count"],
                }

                st.success(f"✅ Processed '{uploaded_file.name}' — {result['upserted_count']} chunks indexed.")

        except PDFLoadError as e:
            st.error(f"PDF error: {e}")
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")

    st.divider()
    st.header("2. Documents in this session")

    if st.session_state.documents:
        for ns, info in st.session_state.documents.items():
            st.write(f"📄 **{info['display_name']}** — {info['chunk_count']} chunks")
    else:
        st.info("No documents processed yet.")

    st.divider()
    st.header("3. Retrieval Settings")

    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=5)
    similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    page_filter_enabled = st.checkbox("Filter by specific page number")
    page_filter_value = None
    if page_filter_enabled:
        page_filter_value = st.number_input("Page number", min_value=1, step=1)

# ============================================================
# MAIN AREA — Query
# ============================================================
if not st.session_state.documents:
    st.warning("👈 Upload and process a PDF from the sidebar to get started.")
else:
    doc_options = {info["display_name"]: ns for ns, info in st.session_state.documents.items()}
    selected_display_name = st.selectbox("Select document to query", list(doc_options.keys()))
    selected_namespace = doc_options[selected_display_name]

    query = st.text_input("Ask a question about the document:")
    ask_clicked = st.button("Ask")

    if ask_clicked:
        if not query or not query.strip():
            st.error("Please enter a question before clicking Ask.")
        else:
            try:
                with st.spinner("Retrieving relevant context..."):
                    retrieved = retrieve_relevant_chunks(
                        query=query,
                        namespace=selected_namespace,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        page_filter=int(page_filter_value) if page_filter_value else None,
                    )

                with st.spinner("Generating answer..."):
                    result = generate_answer(query, retrieved)

                logging.info(f"doc={selected_display_name} | query={query} | grounded={result['grounded']}")

                st.session_state.query_history.append({
                    "question": query,
                    "answer": result["answer"],
                    "document": selected_display_name,
                    "grounded": result["grounded"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                st.subheader("Answer")
                if result["grounded"]:
                    st.success(result["answer"])
                else:
                    st.warning(result["answer"])

                if result["sources"]:
                    st.subheader("📎 Source Attribution")
                    for i, src in enumerate(result["sources"], start=1):
                        with st.expander(f"Source {i} — Page {src['page_number']} (similarity: {src['score']:.3f})"):
                            st.write(src["text"])

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ============================================================
# QUERY HISTORY (mandatory enhancement: session memory)
# ============================================================
if st.session_state.query_history:
    st.divider()
    st.subheader("🕘 Query History")
    for entry in reversed(st.session_state.query_history[-10:]):
        icon = "✅" if entry["grounded"] else "⚠️"
        st.write(f"{icon} **[{entry['timestamp']}] {entry['document']}** — {entry['question']}")
        st.caption(entry["answer"])
