from sentence_transformers import SentenceTransformer
import numpy as np
from src.retrieval.faiss_retrieval import FaissRetriever
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Load query encoder
text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize retriever
retriever = FaissRetriever(
    embedding_path="data/embeddings/text/text_embeddings.npy",
    timestamps_path="data/embeddings/text/timestamps.npy"
)

# Encode query
query = "What is the main topic discussed at the start?"
query_embedding = text_model.encode([query])

# Search
results = retriever.search(query_embedding)

# Print
for ts, score in results:
    print(f"Timestamp: {ts:.2f}s, Score: {score:.4f}")