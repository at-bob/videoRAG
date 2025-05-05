import os
import torch
import clip
import numpy as np
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Frame directory
frames_dir = "data/frames/"
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])

# Process and encode frames
embeddings = []
timestamps = []

for frame_path in frame_files:
    image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    embeddings.append(embedding.cpu().numpy())

    # Extract timestamp from filename (e.g., frame_10s.jpg → 10)
    timestamp = int(os.path.basename(frame_path).split("_")[1].replace("s.jpg", ""))
    timestamps.append(timestamp)

embeddings = np.vstack(embeddings)

# Save embeddings
os.makedirs('data/embeddings/image', exist_ok=True)
np.save('data/embeddings/image/image_embeddings.npy', embeddings)
np.save('data/embeddings/image/timestamps.npy', timestamps)

print("✅ Image embeddings generated and saved.")