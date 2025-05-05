import os
import pytesseract
from PIL import Image
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  

# Paths
frames_dir = "data/frames"
output_dir = "data/embeddings/ocr"
os.makedirs(output_dir, exist_ok=True)

# Load model for embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# OCR + Embedding
ocr_texts = []
timestamps = []
embeddings = []


# Gather frame file paths
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

for fname in tqdm(frame_files, desc="🧠 Running OCR + Embedding"):
    path = os.path.join(frames_dir, fname)
    try:
        text = pytesseract.image_to_string(Image.open(path)).strip()
        if text:
            timestamp = int(fname.split("_")[1].replace("s.jpg", ""))
            embedding = model.encode([text], normalize_embeddings=True)[0]
            ocr_texts.append(text)
            timestamps.append(timestamp)
            embeddings.append(embedding)
    except Exception:
        continue  # skip problematic frames silently

# Save results
np.save(os.path.join(output_dir, "ocr_embeddings.npy"), np.array(embeddings))
np.save(os.path.join(output_dir, "timestamps.npy"), np.array(timestamps))

with open(os.path.join(output_dir, "ocr_texts.json"), "w") as f:
    json.dump({"timestamps": timestamps, "texts": ocr_texts}, f, indent=2)

print("✅ OCR and embedding complete. Outputs saved to data/embeddings/ocr/")