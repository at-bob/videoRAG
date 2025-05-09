===================================================
                VideoRAG - Multimodal Video Retrieval System
===================================================

This project implements a Retrieval-Augmented Generation (RAG) pipeline for video content. 
It enables natural language queries to retrieve relevant video moments using 
transcripts, visual keyframes, and OCR-extracted slide text embeddings.

📌 GitHub Repository: https://github.com/at-bob/videoRAG


==========================================
📁 Project Structure
==========================================

VideoRAG/
├── README.md
├── requirements.txt
├── data/
│   ├── transcripts/       # Stores transcript JSON files
│   ├── frames/            # Extracted video frames as images
│   ├── embeddings/        # Generated embeddings for text, image, and OCR
│   └── video.mp4          # Input video file for processing
├── src/
│   ├── data_processing/   # Scripts for data preprocessing
│   └── retrieval/         # Retrieval models and logic
├── evaluation/
│   ├── gold_test_set.json # Test set with answerable and unanswerable queries
│   └── evaluate_retrieval.py # Evaluation script for model benchmarking
└── streamlit.py           # Interactive UI using Streamlit


==========================================
🚀 Setup & Installation
==========================================

1. Clone the repository:
   git clone https://github.com/at-bob/videoRAG.git
   cd videoRAG

2. Create and activate a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

3. Install required dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install git+https://github.com/openai/CLIP.git

4. (Optional for macOS M1/M2 users)
   brew install openssl
   export LDFLAGS="-L/opt/homebrew/opt/openssl/lib"
   export CPPFLAGS="-I/opt/homebrew/opt/openssl/include"


==========================================
📚 Data Processing Pipeline
==========================================

1. **Extract Frames:**
   Run: `python src/data_processing/extract_frames.py`
   - Extracts video frames every 2 seconds and saves them in `data/frames/`.

2. **Transcribe Audio (Whisper):**
   Run: `python src/data_processing/transcribe.py`
   - Uses OpenAI Whisper to generate transcripts stored in JSON.

3. **Generate Text Embeddings:**
   Run: `python src/data_processing/generate_text_embeddings.py`
   - Generates embeddings from transcripts using MiniLM.

4. **Generate Image Embeddings:**
   Run: `python src/data_processing/generate_image_embeddings.py`
   - Extracts embeddings from keyframes using CLIP.

5. **Generate OCR + Embeddings:**
   Run: `python src/data_processing/ocr_frame_texts.py`
   - Extracts slide text from frames using Tesseract OCR and generates embeddings.

==========================================
🔍 Retrieval Methods
==========================================

- FAISS (Dense Vector Similarity Search)
- PgVector (PostgreSQL Vector Search)
- TF-IDF (Lexical Matching)
- BM25 (Lexical Matching)
- Multimodal (Transcript + Visual Embeddings)
- Multimodal+OCR (Transcript + Visual + Slide Text Embeddings)

To run the evaluation:
```bash
python evaluation/evaluate_retrieval.py
```
==========================================
🎮 Streamlit Interactive UI
Launch the interactive interface:

```bash
streamlit run streamlit.py
```
Enter a natural language query.

Select one of the retrieval models.

View the retrieved timestamp and system confidence.

==========================================
📈 Evaluation Metrics
Top-1 Accuracy: Correct retrievals within a ±10 second tolerance.

Rejection Quality: Ability to reject unanswerable queries.

Latency: Average retrieval time per query.

Evaluation Results can be reviewed in the terminal after running the evaluation script.

==========================================