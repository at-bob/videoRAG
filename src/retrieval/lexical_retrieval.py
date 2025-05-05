import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.tokenize import wordpunct_tokenize
import nltk

# Load transcript segments
with open("data/transcripts/transcript.json", "r") as f:
    transcript = json.load(f)

segments = [seg['text'] for seg in transcript['segments']]
timestamps = [seg['start'] for seg in transcript['segments']]

# === TF-IDF Setup ===
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(segments)

# === BM25 Setup ===
tokenized_corpus = [wordpunct_tokenize(doc.lower()) for doc in segments]
bm25 = BM25Okapi(tokenized_corpus)

# # === User Query ===
# query = input("🔍 Enter your question: ")

# # --- TF-IDF Retrieval ---
# query_vec = vectorizer.transform([query])
# tfidf_scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
# top_tfidf = np.argsort(tfidf_scores)[::-1][:5]

# print("\n📚 Top TF-IDF Matches:")
# for i in top_tfidf:
#     print(f"⏱️ {timestamps[i]:.2f}s — Score: {tfidf_scores[i]:.4f}")

# # --- BM25 Retrieval ---
# tokenized_query = wordpunct_tokenize(query.lower())
# bm25_scores = bm25.get_scores(tokenized_query)
# top_bm25 = np.argsort(bm25_scores)[::-1][:5]

# print("\n📚 Top BM25 Matches:")
# for i in top_bm25:
#     print(f"⏱️ {timestamps[i]:.2f}s — Score: {bm25_scores[i]:.4f}")

def run_tfidf_query(query, return_if_none=False):
    query_vec = vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
    best_idx = np.argmax(tfidf_scores)
    if tfidf_scores[best_idx] < 0.4:
        return None if return_if_none else 0.0
    return timestamps[best_idx]

def run_bm25_query(query, return_if_none=False):
    tokenized_query = wordpunct_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    best_idx = np.argmax(bm25_scores)
    if bm25_scores[best_idx] < 8.5: 
        return None if return_if_none else 0.0
    return timestamps[best_idx]