import os
import json
import numpy as np
import openai
import faiss
from dotenv import load_dotenv

# Load environment variables from .env (if used)
load_dotenv()

# Set up the OpenAI client (use your proxy token and API base)
openai.api_key = os.getenv('AIPROXY_TOKEN')
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1" 

# Define file paths for the report rules document and index files.
REPORT_RULES_FILE = "report_rules.txt"         # The document with report guidelines/rules.
REPORT_INDEX_FILE = "report_index.bin"           # The FAISS index will be saved here.
REPORT_CHUNKS_FILE = "report_chunks.json"        # The list of document chunks will be saved here.

def load_report_chunks(file_path, delimiter="\n\n"):
    """
    Reads the report rules file and splits it into chunks based on the delimiter.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Split text by double newlines (paragraphs) and remove extra whitespace.
        chunks = [chunk.strip() for chunk in text.split(delimiter) if chunk.strip()]
        return chunks
    except Exception as e:
        print(f"Error loading report rules document: {e}")
        return []

def get_embedding(text):
    """
    Gets an embedding for the given text using the "text-embedding-3-small" model.
    """
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
    """
    Computes embeddings for each chunk and builds a FAISS index.
    Returns the index and the embeddings as a NumPy array.
    """
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb is not None:
            embeddings.append(emb)
        else:
            # Fallback: use a zero vector; adjust the dimension as needed (e.g., 768)
            embeddings.append(np.zeros(768))
    embeddings_array = np.array(embeddings).astype("float32")
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)
    return index, embeddings_array

def main():
    # Load the report rules document and split it into chunks.
    chunks = load_report_chunks(REPORT_RULES_FILE)
    if not chunks:
        print("Error: No report rules found in the document.")
        return

    # Build the FAISS index.
    index, embeddings_array = build_faiss_index(chunks)
    
    # Save the FAISS index to disk.
    faiss.write_index(index, REPORT_INDEX_FILE)
    print(f"FAISS index saved to {REPORT_INDEX_FILE}")
    
    # Save the document chunks to a JSON file.
    with open(REPORT_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print(f"Document chunks saved to {REPORT_CHUNKS_FILE}")

if __name__ == "__main__":
    main()