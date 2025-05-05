import numpy as np
import faiss

class FaissRetriever:
    def __init__(self, embedding_path, timestamps_path):
        # Load embeddings and timestamps
        self.embeddings = np.load(embedding_path)
        self.timestamps = np.load(timestamps_path)
        
        # Normalize embeddings (important for cosine similarity)
        faiss.normalize_L2(self.embeddings)
        
        # Build index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # IP = Inner Product (cosine similarity after normalization)
        self.index.add(self.embeddings)
    
    def search(self, query_embedding, top_k=5):
        # Normalize query
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, top_k)
        
        results = [(self.timestamps[idx], float(D[0][i])) for i, idx in enumerate(I[0])]
        return results