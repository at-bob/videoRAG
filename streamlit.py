import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_NO_PARALLEL"] = "1"

import multiprocessing as mp
mp.set_start_method("fork", force=True)

import streamlit as st
import time

from sentence_transformers import SentenceTransformer
import numpy as np

# ── ENABLE ONE RETRIEVER AT A TIME ─────────────────────────────────────────────
# 1) FAISS
from src.retrieval.faiss_retrieval import FaissRetriever

EMBEDDING_PATH  = "/Users/adamtai/Desktop/GPCS/3. SPRING 2025/CMPS 396AH/Assignments/Assignment 5/multimodal-RAG/data/embeddings/text/text_embeddings.npy"
TIMESTAMPS_PATH = "/Users/adamtai/Desktop/GPCS/3. SPRING 2025/CMPS 396AH/Assignments/Assignment 5/multimodal-RAG/data/embeddings/text/timestamps.npy"

assert os.path.isfile(EMBEDDING_PATH), f"No file at {EMBEDDING_PATH}"
assert os.path.isfile(TIMESTAMPS_PATH), f"No file at {TIMESTAMPS_PATH}"

retriever = FaissRetriever(embedding_path=EMBEDDING_PATH, timestamps_path=TIMESTAMPS_PATH)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def _faiss(query: str):
    # 1) Embed and reshape
    vec = embedder.encode([query], convert_to_numpy=True)[0].astype(np.float32).reshape(1, -1)
    # 2) Do the search
    out = retriever.search(vec, top_k=1)
    # 3) Extract timestamp
    ts = out[0][0]
    if ts == 0:
        return None
    return ts

faiss = _faiss

# 2) pgvector
from src.retrieval.pgvector_query import run_pgvector_query
def pgvector_retrieve(query: str):
    ts = run_pgvector_query(query)
    if ts == 0:
        return None
    return ts
    
# 3) TF‑IDF
from src.retrieval.lexical_retrieval import run_tfidf_query
def tfidf_retrieve(query: str):
    results = run_tfidf_query(query)      
    if not results:
        return None
    ts = results[0]
    if ts == 0:
        return None
    return ts

# 4) BM25
from src.retrieval.lexical_retrieval import run_bm25_query
def bm25_retrieve(query: str):
    results = run_bm25_query(query)
    if not results:
        return None
    ts = results[0]
    if ts == 0:
        return None
    return ts

# 5) Multimodal
from src.retrieval.multimodal_retrieval import run_multimodal_query
def multimodal_retrieve(query: str):
    ts = run_multimodal_query(query)
    if ts == 0:
        return None
    return ts

# 6) Multimodal + OCR
def multimodal_ocr_retrieve(query: str):
    ts = run_multimodal_query(query, alpha=0.6, beta=0.2, gamma=0.2)
    if ts == 0:
        return None
    return ts

# ── STREAMLIT UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Tester", layout="wide")
st.title("🔎 Retrieval Module Tester")

VIDEO_URL = "https://www.youtube.com/watch?v=dARr3lGKwk8"

query = st.text_input("Enter your query:")

method = st.selectbox(
    "Choose retrieval method:",
    ["FAISS", "pgvector", "TFDIF", "BM25", "Multimodal", "Multimodal+OCR"] 
)

if st.button("Run"):
    if not query:
        st.warning("Please enter a query.")
    else:
        start = time.time()

        # Dispatch to the correct wrapper
        if method == "FAISS":
            ts = faiss(query)
        elif method == "pgvector":
            ts = pgvector_retrieve(query)
        elif method == "TF‑IDF":
            ts = tfidf_retrieve(query)
        elif method == "BM25":
            ts = bm25_retrieve(query)
        elif method == "Multimodal":
            ts = multimodal_retrieve(query)
        elif method == "Multimodal+OCR":
            ts = multimodal_ocr_retrieve(query)
        else:
            ts = None

        # Show result or “not found”
        if ts is None:
            st.error("❌ Answer not found in the scope of this video.")
        else:
            st.success(f"📍 Found at {ts:.2f}s")
            # Display the video at that timestamp
            try:
                st.video(VIDEO_URL, start_time=int(ts))
            except TypeError:
                # fallback if your Streamlit version doesn’t support start_time
                st.markdown(f"""
                <video width="700" controls>
                  <source src="{VIDEO_URL}#t={int(ts)}" type="video/mp4">
                </video>""", unsafe_allow_html=True)

        st.write(f"⏱ Retrieval time: {time.time() - start:.3f}s")