# greene-mentor

A Retrieval-Augmented Generation (RAG) chatbot inspired by Robert Greene's works.  
Powered by FastAPI, MiniLM embeddings, and Groq's Llama-3.1-8b-instant.

## Features
- `/ask` → Pure Groq completion in Greene-like style
- `/ask_rag` → Retrieval-augmented answers with transcript citations

## Setup
```bash
git clone https://github.com/<your-username>/greene-mentor-api.git
cd greene-mentor-api
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
