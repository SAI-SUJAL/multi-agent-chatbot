
from locale import normalize
import os
import fitz  # PyMuPDF
import faiss  # FAISS for vector search
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import openai
import pickle  # For saving FAISS index
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# Initialize FAISS and Embeddings Model
faiss_index_path = "faiss_index.bin"
text_mapping_path = "text_mappings.pkl"

embedding_dim = 384  # Must match MiniLM model

# Check if FAISS index exists
if os.path.exists(faiss_index_path) and os.path.exists(text_mapping_path):
    print("🔄 Loading FAISS index from file...")
    index = faiss.read_index(faiss_index_path)
    with open(text_mapping_path, "rb") as f:
        id_to_text = pickle.load(f)
    print(f"✅ FAISS Index Successfully Loaded! Index Size: {index.ntotal}")
else:
    print("⚠️ No FAISS index found. Creating a new one...")
    index = faiss.IndexFlatIP(embedding_dim)  # Using cosine similarity
    id_to_text = {}
# Extract text from PDFs page by page
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_text = []
    for page_num in range(len(doc)):  # Iterate over each page
        text = doc[page_num].get_text("text").strip()
        if text:
            extracted_text.append(f"Page {page_num + 1}: {text}")
    return extracted_text


def load_pdfs(pdf_paths):
    global index, id_to_text
    doc_id = 0

    for pdf_path in pdf_paths:
        print(f"📄 Loading PDF: {pdf_path}")
        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            print(f"⚠️ No text extracted from {pdf_path}!")
            continue

        for page_text in pages:
            print(f"✅ Storing in FAISS: {page_text[:200]}")  # Print first 200 chars
            embedding = normalize_vector(embedder.encode(page_text).astype(np.float32).reshape(1, -1))
            index.add(embedding)
            id_to_text[doc_id] = page_text
            doc_id += 1

    # Save FAISS index
    faiss.write_index(index, "faiss_index.bin")
    with open("text_mappings.pkl", "wb") as f:
        pickle.dump(id_to_text, f)

    print(f"📌 FAISS Index Size After Loading: {index.ntotal}")

    
#     return results
def query_agent(user_query, top_k=5):  # Increase top_k to get more results
    query_embedding = normalize_vector(embedder.encode(user_query).astype(np.float32).reshape(1, -1))
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx in id_to_text and distance < 1.0:  # Filter strong matches
            print(f"✅ FAISS Match (Score: {distance}): {id_to_text[idx][:200]}")
            results.append(id_to_text[idx])

    if not results:
        print("⚠️ No strong matches found in FAISS!")
        return ["No strong matches found in legal database."]

    return results


# Summarization Agent using Groq API
def summarization_agent(texts):
    prompt = f"Summarize the following legal texts in simple terms:\n{texts}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5
    )

    return response["choices"][0]["message"]["content"]
  

# FastAPI Setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
@app.post("/ask")
def ask_legal_bot(request: QueryRequest):
    try:
        relevant_texts = query_agent(request.query)
        if not relevant_texts:
            return {"summary": "No relevant legal information found."}
        print(f"📌 FAISS Index Size: {index.ntotal}")


        summary = summarization_agent(" ".join(relevant_texts))
        return {"summary": summary}

    except Exception as e:
        print("🔥 API Error:", str(e))  # Log the error
        return {"error": "Internal Server Error. Check logs for details."}

def normalize_vector(vec):  # Renamed function to avoid conflicts
    return vec / np.linalg.norm(vec)  # Normalize to unit length

# Modify embedding storage
def test_faiss_query():
    test_query = "steps to file a lawsuit"
    query_embedding = normalize_vector(embedder.encode(test_query).astype(np.float32).reshape(1, -1))

    distances, indices = index.search(query_embedding, 3)

    print("\n📌 FAISS Manual Test Results:")
    for idx in indices[0]:
        if idx in id_to_text:
            print(f"🔍 Retrieved: {id_to_text[idx][:200]}")  # First 200 characters
        else:
            print(f"❌ FAISS returned an invalid index: {idx}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_DIR = os.path.join(BASE_DIR, "data")  # Store PDFs inside "data/" folder

# Define PDF paths dynamically
    pdfs = [
    os.path.join(PDF_DIR, "Guide_to_Litigation.pdf"),
    os.path.join(PDF_DIR, "Legal_Compliance.pdf"),
    ]
    
    load_pdfs(pdfs)  # Process PDFs first

    print(f"✅ PDFs processed! FAISS index size: {index.ntotal}")
    
    # Run FAISS test
    # test_faiss_query()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=True)
