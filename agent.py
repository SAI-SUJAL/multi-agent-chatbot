import os
import streamlit as st
from dotenv import load_dotenv

# Phi Imports
from phi.agent import Agent
from phi.llm.groq import Groq  # Updated import

# Langchain Imports
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Groq Imports
from groq import Groq as GroqClient

# Other Imports
import traceback

# Load environment variables
load_dotenv()

# Configuration and Setup
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
if not GROQ_API_KEY:
    st.error("Groq API Key must be set in .env file")

# Vector Database Initialization
def initialize_vector_db(pdf_paths):
    """Initialize FAISS vector database from PDF documents."""
    documents = []
    for file_path in pdf_paths:
        if os.path.exists(file_path):
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Error loading {file_path}: {e}")
        else:
            st.warning(f"PDF file not found: {file_path}")

    if not documents:
        st.error("No documents could be loaded")
        return None

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Embedding Model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS Vector Store
    return FAISS.from_documents(texts, embedding_model)

# PDF Paths Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "backend", "data")

# Define PDF paths dynamically
PDF_PATHS = [
    os.path.join(PDF_DIR, "Guide_to_Litigation.pdf"),
    os.path.join(PDF_DIR, "Legal_Compliance.pdf"),
]

# Main Streamlit App
def main():
    st.title("AI Legal Assistant")
    st.subheader("Advanced Legal Information Retrieval")

    # Validate API Key
    if not GROQ_API_KEY:
        st.error("Please set your Groq API Key in the .env file")
        return

    # Initialize Vector Database
    vector_db = initialize_vector_db(PDF_PATHS)
    
    if vector_db is None:
        st.error("Could not initialize vector database. Check your PDF files.")
        return

    # Rest of your existing main function code remains the same...

if __name__ == "__main__":
    main()
