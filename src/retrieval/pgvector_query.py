from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import numpy as np

# Replace with your credentials or use getpass.getuser()
DB_URL = "postgresql://adamtai@localhost/postgres"

# Load the same embedding model used earlier
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# User query
# query = input("🔍 Enter your question: ")
# query_embedding = model.encode([query])[0].tolist()

# Connect to the DB and run the vector similarity query
# engine = create_engine(DB_URL)
# with engine.connect() as conn:
#     results = conn.execute(text("""
#         SELECT timestamp, embedding <-> (:query_vector)::vector AS score
#         FROM text_embeddings
#         ORDER BY embedding <-> (:query_vector)::vector
#         LIMIT 5
#     """), {"query_vector": query_embedding}).fetchall()

# Display results
# print("\n🧠 Top matching timestamps:")
# for ts, score in results:
#     print(f"⏱️ {ts:.2f}s — Similarity: {score:.4f}")

def run_pgvector_query(query, return_if_none=False):
    from sqlalchemy import create_engine, text
    from sentence_transformers import SentenceTransformer

    DB_URL = "postgresql://adamtai@localhost/postgres"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0].tolist()

    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT timestamp, embedding <-> (:query_vector)::vector AS score
            FROM text_embeddings
            ORDER BY embedding <-> (:query_vector)::vector
            LIMIT 1
        """), {"query_vector": query_embedding}).fetchone()

    if result is None:
        return None if return_if_none else 0.0

    timestamp, distance = result
    if distance > 0.73: 
        return None if return_if_none else 0.0
    return timestamp
