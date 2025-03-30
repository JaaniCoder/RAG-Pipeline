# Retrieval-Augmented Generation (RAG) Notebook

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using OpenAI's API and sentence transformers. The notebook extracts text from PDF documents, creates embeddings, performs semantic search, and generates responses based on user queries.

## Features
- Extracts text from PDFs using `PyMuPDF (fitz)`
- Chunks text into manageable segments
- Generates embeddings using `Sentence Transformers`
- Computes cosine similarity to find relevant chunks
- Uses OpenAI API to generate responses based on retrieved text

## Installation
Ensure you have the required dependencies installed:
```bash
pip install torch sentence-transformers pymupdf openai numpy
```

## Usage
1. **Extract text from a PDF:**
   ```python
   text = extract_text_from_pdf("document.pdf")
   ```

2. **Chunk the text:**
   ```python
   chunks = chunk_text(text, chunk_size=512)
   ```

3. **Generate embeddings:**
   ```python
   embeddings = create_embeddings(chunks)
   ```

4. **Perform semantic search:**
   ```python
   result = semantic_search(query, embeddings, chunks)
   ```

5. **Generate a response:**
   ```python
   response = generate_response(query, result)
   ```

## Functions
- `extract_text_from_pdf(pdf_path)`: Extracts text from a given PDF file.
- `chunk_text(text, chunk_size)`: Splits text into smaller parts.
- `create_embeddings(text_chunks)`: Generates embeddings for text chunks.
- `cosine_similarity(vec1, vec2)`: Computes similarity between vectors.
- `semantic_search(query, embeddings, text_chunks)`: Finds the most relevant chunk.
- `generate_response(query, retrieved_text)`: Uses OpenAI API to generate a response.

## Dependencies
- Python
- `torch`
- `sentence-transformers`
- `pymupdf`
- `openai`
- `numpy`

## License
This project is licensed under the MIT License.

