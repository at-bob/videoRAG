import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Load transcript
with open("data/transcripts/transcript.json", "r") as f:
    transcript = json.load(f)

segments = [seg['text'] for seg in transcript['segments']]
timestamps = [seg['start'] for seg in transcript['segments']]

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(segments, convert_to_numpy=True)

# Save embeddings
os.makedirs('data/embeddings/text', exist_ok=True)
np.save('data/embeddings/text/text_embeddings.npy', embeddings)
np.save('data/embeddings/text/timestamps.npy', timestamps)

print("✅ Text embeddings generated and saved.")