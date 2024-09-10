# Conversational AI with PDF Content Retrieval

This project sets up a conversational AI chatbot using LangChain and Hugging Face, which answers questions based on content extracted from a PDF document. The system leverages both generative and retrieval-based models to provide relevant responses.

## Features

- **Conversational AI**: Powered by `DialoGPT-small` for interactive dialogue generation.
- **PDF Content Extraction**: Utilizes `pdfplumber` to extract text from PDF documents.
- **Retrieval-Based QA**: Employs `sentence-transformers/all-MiniLM-L6-v2` for text embeddings and FAISS for efficient vector search.

## Prerequisites

Before setting up the project, ensure you have:

- Python 3.8 or later installed.
- A Hugging Face account (optional, for accessing more models).

## Installation

Follow these steps to set up and run the project:

### 1. Clone the Repository

Clone the project repository from GitHub:

```bash
git clone 
cd your-repo

python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

pip install -r requirements.txt

