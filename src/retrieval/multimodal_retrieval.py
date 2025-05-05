import numpy as np
from sentence_transformers import SentenceTransformer
import clip
import torch
from PIL import Image

# === Load OCR Text Embeddings ===
ocr_embeddings = np.load("data/embeddings/ocr/ocr_embeddings.npy")
ocr_timestamps = np.load("data/embeddings/ocr/timestamps.npy")

# === Load Text Embeddings ===
text_embeddings = np.load("data/embeddings/text/text_embeddings.npy")
text_timestamps = np.load("data/embeddings/text/timestamps.npy")

# === Load Image Embeddings ===
image_embeddings = np.load("data/embeddings/image/image_embeddings.npy")
image_timestamps = np.load("data/embeddings/image/timestamps.npy")

# === Normalize all embeddings for cosine similarity ===
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

text_embeddings = normalize(text_embeddings)
image_embeddings = normalize(image_embeddings)
ocr_embeddings = normalize(ocr_embeddings)

# === Load Models ===
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# === Query Function ===
def is_visual_query(query):
    visual_keywords = ["show", "picture", "scene", "look", "diagram", "illustration", "see"]
    return any(word in query.lower() for word in visual_keywords)

def run_multimodal_query(query, return_if_none=False, alpha=None, beta=None, gamma=None):
    """
    alpha = weight for text transcript score
    beta = weight for keyframe visual (CLIP) score
    gamma = weight for OCR slide text score
    """
    # Default weighting (can be tuned later)
    if alpha is None or beta is None or gamma is None:
        if is_visual_query(query):
            alpha, beta, gamma = 0.3, 0.3, 0.4
        else:
            alpha, beta, gamma = 0.6, 0.2, 0.2

    # --- Textual query embedding (MiniLM) ---
    query_text_emb = text_model.encode([query], normalize_embeddings=True)[0]

    # --- Visual embedding (CLIP) ---
    with torch.no_grad():
        clip_query = clip.tokenize([query]).to(device)
        query_img_emb = clip_model.encode_text(clip_query).cpu().numpy()[0]
    query_img_emb = query_img_emb / np.linalg.norm(query_img_emb)

    # --- Compute similarities ---
    text_scores = np.dot(text_embeddings, query_text_emb)
    image_scores = np.dot(image_embeddings, query_img_emb)
    ocr_scores = np.dot(ocr_embeddings, query_text_emb)  # OCR is text, same embedding

    # --- Fuse scores across all 3 modalities ---
    fused_scores = []
    for i, ts in enumerate(text_timestamps):
        img_idx = np.argmin(np.abs(image_timestamps - ts))
        ocr_idx = np.argmin(np.abs(ocr_timestamps - ts))

        score = (
            alpha * text_scores[i] +
            beta * image_scores[img_idx] +
            gamma * ocr_scores[ocr_idx]
        )
        fused_scores.append((ts, score))

    fused_scores.sort(key=lambda x: x[1], reverse=True)
    best_ts, best_score = fused_scores[0]

    if best_score < 0.45:
        return None if return_if_none else 0.0

    return best_ts
