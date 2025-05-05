# Video RAG

An end-to-end multimodal retrieval-augmented generation (RAG) system that answers user queries by retrieving relevant video segments based on both audio and visual embeddings.

## Project Structure

```plaintext
multimodal-rag-video-qa/
├── README.md
├── requirements.txt
├── data/
│   ├── transcripts/
│   ├── frames/
│   └── embeddings/
├── src/
│   ├── data_processing/
│   ├── embeddings/
│   ├── retrieval/
│   └── app/
├── evaluation/
│   ├── gold_test_set.json
│   └── evaluation_results.ipynb
└── docs/
    └── report.md
```

## Setup Instructions
```bash
pip install -r requirements.txt
