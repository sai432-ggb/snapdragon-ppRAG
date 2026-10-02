![Python](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
# Privacy-Preserving Retrieval-Augmented Generation (ppRAG)

This repository implements an end-to-end **Privacy-Preserving Retrieval-Augmented Generation (ppRAG)** pipeline.  
It combines **AES-256 payload encryption**, **CAPRISE vector encryption**, and **DistanceDP differential privacy** to enable secure document ingestion, encrypted cloud storage, and private query retrieval — all while generating answers locally with Ollama.

---

## System Architecture

![Architecture Diagram](pprag_architecture_diagram.png)

**Stage 1: Local Ingestion & Encryption**  
- Documents are embedded locally.  
- AES-256 encrypts raw text.  
- CAPRISE encrypts embeddings before upload.  

**Stage 2: Secure Cloud Search**  
- Queries are perturbed with DistanceDP noise.  
- CAPRISE encrypts query embeddings.  
- Cloud database compares encrypted vectors only.  

**Stage 3: Local Decryption & Generation**  
- Candidate vectors are decrypted locally.  
- AES payloads are decrypted client-side.  
- Context is passed to Ollama for private inference.  

---

## Features

- AES-256 Payload Encryption  
- CAPRISE Vector Encryption (distance-preserving)  
- DistanceDP Query Perturbation  
- Local Decryption & Re-Ranking  
- Local LLM Inference with Ollama  

---

## Project Structure

- `app.py` – Streamlit dashboard for ingestion, search, and LLM generation.  
- `pprag_master_system_v2.py` – Core ppRAG pipeline implementation.  
- `pprag_architecture_diagram.png` – Architecture flowchart.  
- `LICENSE` – MIT License.  
- `README.md` – Documentation.  

---

## Getting Started

### Prerequisites
- Python 3.12+
- Ollama installed locally
- Virtual environment (`venv`) activated

### Install Dependencies
```bash
pip install cryptography numpy streamlit
Run the Dashboard
streamlit run app.py
Run the Core Pipeline


###Why This Matters
Traditional RAG pipelines risk data leakage when embeddings or queries are stored in cloud databases.
This ppRAG pipeline solves that by:

Encrypting sensitive text and vectors before upload.

Preserving retrieval accuracy while hiding internal relationships.

Ensuring only your local machine ever sees decrypted content.

It is a Zero-Trust RAG system: the cloud sees only ciphertexts, never your secrets.


Pull requests are welcome.
If you’d like to extend this pipeline (e.g., connect to FAISS/ChromaDB, add new LLM backends), please open an issue or PR.
