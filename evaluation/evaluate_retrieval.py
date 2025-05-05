import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from src.retrieval.faiss_retrieval import FaissRetriever
from src.retrieval.pgvector_query import run_pgvector_query
from src.retrieval.lexical_retrieval import run_tfidf_query, run_bm25_query
from src.retrieval.multimodal_retrieval import run_multimodal_query


# === CONFIG ===
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
TOLERANCE_SECONDS = 10.0  # How close a match must be to count as correct

# Load gold test set
with open("evaluation/gold_test_set.json") as f:
    gold = json.load(f)

answerable = gold["answerable"]
unanswerable = gold["unanswerable"]

# Init FAISS
faiss_retriever = FaissRetriever(
    "data/embeddings/text/text_embeddings.npy",
    "data/embeddings/text/timestamps.npy"
)

# === Evaluation functions ===
def evaluate_top1_accuracy(questions, retrieval_fn):
    correct = 0
    total_time = 0
    for item in questions:
        start = time.time()
        top1_timestamp = retrieval_fn(item["question"])
        total_time += time.time() - start
        if abs(top1_timestamp - item["timestamp"]) <= TOLERANCE_SECONDS:
            correct += 1
    return correct / len(questions), total_time / len(questions)

def evaluate_rejection_quality(questions, retrieval_fn):
    false_positives = 0
    total_time = 0
    for item in questions:
        start = time.time()
        result = retrieval_fn(item["question"], return_if_none=True)
        total_time += time.time() - start
        if result is not None:  # Should return None or low confidence if not found
            false_positives += 1
    return 1 - (false_positives / len(questions)), total_time / len(questions)

# === Retrieval wrappers ===
def faiss_wrapper(query, return_if_none=False):
    embedding = EMBEDDING_MODEL.encode([query])
    results = faiss_retriever.search(embedding)
    score = results[0][1] if results else 0.0
    if score < 0.45:  
        return None if return_if_none else 0.0
    return results[0][0]

def pgvector_wrapper(query, return_if_none=False):
    return run_pgvector_query(query, return_if_none=return_if_none)

def tfidf_wrapper(query, return_if_none=False):
    return run_tfidf_query(query, return_if_none=return_if_none)

def bm25_wrapper(query, return_if_none=False):
    return run_bm25_query(query, return_if_none=return_if_none)

def multimodal_wrapper(query, return_if_none=False):
    return run_multimodal_query(query, return_if_none=return_if_none, alpha=0.6, beta=0.4)

def multimodal_ocr_wrapper(query, return_if_none=False):
    # Uses 3-modality fusion: transcript + keyframe + OCR
    return run_multimodal_query(query, return_if_none=return_if_none)

# === Run evaluations ===
methods = {
    "FAISS": faiss_wrapper,
    "pgvector": pgvector_wrapper,
    "TF-IDF": tfidf_wrapper,
    "BM25": bm25_wrapper,
    "Multimodal": multimodal_wrapper,
    "Multimodal+OCR": multimodal_ocr_wrapper
}

print("🔍 Evaluation Results:\n")
for name, fn in methods.items():
    acc, acc_time = evaluate_top1_accuracy(answerable, fn)
    rej, rej_time = evaluate_rejection_quality(unanswerable, fn)
    print(f"🔹 {name}")
    print(f"   ✅ Top-1 Accuracy:        {acc*100:.1f}%")
    print(f"   🚫 Rejection Quality:    {rej*100:.1f}%")
    print(f"   ⏱️ Avg Latency:          {(acc_time + rej_time)/2:.3f}s\n")