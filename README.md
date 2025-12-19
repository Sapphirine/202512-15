# 202512-15 6893proj--RAG-Based-Interview-Assistant-for-IC-Design-Verification
CUEE 6893 course project, ID:202512-15
# RAG-Based Interview Assistant for IC Design & Verification

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot designed to assist students and engineers preparing for **digital IC design and verification interviews**. The system retrieves information from curated technical PDFs and generates grounded, domain-specific answers using a local large language model.

---

## Overview

The chatbot combines document retrieval with large language model inference to provide accurate and context-aware responses. Unlike cloud-based chatbots, this system is **offline-capable** and runs entirely on local hardware.

**Key features:**
- Domain-specific question answering for IC design & verification
- Retrieval-augmented generation to minimize hallucination
- Local LLM inference using vLLM
- Lightweight front-end for interactive querying
- No dependency on external APIs or internet access

---

## System Architecture

The system consists of the following components:

1. **Document Ingestion**
   - Technical PDFs (textbooks, lecture notes, interview materials)
   - Loaded and split into overlapping text chunks

2. **Embedding & Retrieval**
   - Embeddings generated using Qwen3-Embedding-0.6B
   - FAISS vector database for efficient similarity search

3. **LLM Inference Backend**
   - Qwen2.5–3B-Instruct served via vLLM
   - Low-latency, GPU-accelerated inference
   - Answers strictly grounded in retrieved context

4. **Front-End Website**
   - Browser-based interface for user queries
   - Displays model responses interactively
   - Some UI components are reserved for future expansion

---

## Requirements

- Python 3.9+
- NVIDIA GPU (consumer-grade GPUs supported, e.g., RTX 30/40 series)
- CUDA-compatible environment

Main dependencies:
- LangChain
- FAISS
- vLLM
- PyTorch
- Sentence Transformers

---

## How It Works

1. PDFs are preprocessed and indexed offline.
2. User submits a query via the website or script.
3. The query is embedded and matched against the FAISS index.
4. Top relevant text chunks are retrieved.
5. The LLM generates an answer strictly based on retrieved content.
6. If no relevant information is found, the system responds with an explicit refusal.


 **Website**  
https://mgx.dev/app/35c1145ed04d4ec9aa39758dedc87954

---

## Team

- Mingzhi Li  
- Xinyu Liu  
- Qianxu Fu  

Course Project for **CSEE 6893 – Big Data Analytics**, Fall 2025

---

## License

This project is intended for academic and educational use.

