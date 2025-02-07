import os
import json
import numpy as np
import faiss
import openai

# Make sure your OpenAI credentials are set up (hard-coded for testing, but ideally use environment variables)
openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDE0NDdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9._NZRQhxfqSjUJMWcKcfht63t0G35hRFbScM006IYz_M"
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

# Path to your document
DEFAULT_FILE_PATH = "default.txt"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "document_chunks.json"

def load_document_chunks(file_path, delimiter="\n\n"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = [chunk.strip() for chunk in text.split(delimiter) if chunk.strip()]
        return chunks
    except Exception as e:
        print(f"Error loading document: {e}")
        return []

def get_embedding(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding for text: {e}")
        return None

def build_faiss_index(chunks):
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb is not None:
            embeddings.append(emb)
        else:
            embeddings.append(np.zeros(768))  # Adjust dimension if necessary.
    embeddings_array = np.array(embeddings).astype("float32")
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)
    return index, embeddings_array

if __name__ == "__main__":
    chunks = load_document_chunks(DEFAULT_FILE_PATH)
    if not chunks:
        print("Error: No chunks found in the document.")
        exit(1)
    index, embeddings_array = build_faiss_index(chunks)
    # Save the FAISS index to disk.
    faiss.write_index(index, INDEX_FILE)
    # Save the document chunks to disk.
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print("Index and chunks saved successfully.")