import os
import streamlit as st
from dotenv import load_dotenv

# Phi Imports
from phi.agent import Agent
from phi.llm.groq import Groq  # Updated import

# Langchain Imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Groq Imports
from groq import Groq as GroqClient  # Direct Groq client import

# Other Imports
import traceback

# Load environment variables


# Configuration and Setup
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("Groq API Key must be set in .env file")

# Vector Database Initialization
def initialize_vector_db(pdf_paths):
    """Initialize FAISS vector database from PDF documents."""
    documents = []
    for file_path in pdf_paths:
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

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

# PDF Paths (replace with your actual PDF paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "backend/data")  # Store PDFs inside "data/" folder

# Define PDF paths dynamically
PDF_PATHS = [
    os.path.join(PDF_DIR, "Guide_to_Litigation.pdf"),
    os.path.join(PDF_DIR, "Legal_Compliance.pdf"),
]

# Initialize Vector Database
try:
    vector_db = initialize_vector_db(PDF_PATHS)
except Exception as e:
    print(f"Vector DB Initialization Error: {e}")
    vector_db = None

# Initialize Groq Client
groq_client = GroqClient(api_key=GROQ_API_KEY)

# Utility Functions for Agents
def retrieve_documents(query, vector_db):
    """Retrieve documents from vector database"""
    if vector_db is None:
        return "Vector database not initialized."
    
    try:
        docs = vector_db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error retrieving documents: {e}"

def simplify_legal_text(text):
    """Simplify legal text using LLM"""
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a legal expert who simplifies complex legal text for a non-lawyer audience."},
                {"role": "user", "content": f"Simplify this legal text:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error simplifying text: {e}"

# Multi-Agent Workflow Coordinator
class LegalAssistantWorkflow:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        
    def process_query(self, query):
        # Step 1: Refine Query
        try:
            refined_query_response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert at refining and clarifying legal queries."},
                    {"role": "user", "content": f"Rephrase and clarify this legal query to ensure precise understanding:\n\n{query}"}
                ]
            )
            refined_query = refined_query_response.choices[0].message.content
        
            # Step 2: Retrieve Documents
            retrieved_docs = retrieve_documents(refined_query, self.vector_db)
            
            # Step 3: Summarize Retrieved Documents
            summarized_content = simplify_legal_text(retrieved_docs)
            
            return {
                "refined_query": refined_query,
                "retrieved_documents": retrieved_docs,
                "simplified_summary": summarized_content
            }
        except Exception as e:
            return {
                "refined_query": "Error in query processing",
                "retrieved_documents": f"Error: {str(e)}",
                "simplified_summary": f"An error occurred: {str(e)}"
            }

# Streamlit Interface
def main():
    st.title("AI Legal Assistant")
    st.subheader("Advanced Legal Information Retrieval")

    # Workflow Initialization
    if vector_db is None:
        st.error("Vector database could not be initialized. Please check your PDF files.")
        return

    workflow = LegalAssistantWorkflow(vector_db)

    # Query Input
    query = st.text_area(
        "Enter your legal query:", 
        height=150,
        placeholder="Describe your legal question or concern..."
    )

    if st.button("Get Legal Insights"):
        if not query:
            st.warning("Please enter a legal query")
            return

        try:
            # Process Query
            result = workflow.process_query(query)

            # Display Results
            st.subheader("Query Analysis")
            st.markdown(f"**Refined Query:** {result['refined_query']}")

            st.subheader("Retrieved Legal Documents")
            st.text_area("Retrieved Documents", value=result['retrieved_documents'], height=200)

            st.subheader("Simplified Legal Summary")
            st.markdown(result['simplified_summary'])

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
