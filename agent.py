import os
import streamlit as st
import traceback

# Updated Langchain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Groq Imports
from groq import Groq as GroqClient  # Direct Groq client import

# Configuration
PDF_DIR = "backend/data"  # Consistent PDF directory

# Retrieve API Key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq Client
groq_client = GroqClient(api_key=GROQ_API_KEY)

# Utility Functions
def initialize_vector_db(pdf_paths):
    """Initialize FAISS vector database from PDF documents."""
    documents = []
    for file_path in pdf_paths:
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")

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

def retrieve_documents(query, vector_db, k=5):
    """Retrieve documents from vector database."""
    if vector_db is None:
        return "Vector database not initialized."
    
    try:
        docs = vector_db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error retrieving documents: {e}"

def simplify_legal_text(text):
    """Simplify legal text using LLM."""
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
        try:
            # Step 1: Refine Query
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

# Streamlit UI (Simplified)
def main():
    st.title("⚖️ AI Legal Assistant")
    st.subheader("Retrieve and Understand Legal Information")

    # PDF Paths
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        PDF_PATHS = [
            os.path.join(base_dir, PDF_DIR, "Guide_to_Litigation.pdf"),
            os.path.join(base_dir, PDF_DIR, "Legal_Compliance.pdf"),
        ]
    except Exception as e:
        st.error(f"Error finding PDF files: {e}")
        return

    # Vector Database Initialization
    try:
        vector_db = initialize_vector_db(PDF_PATHS)
    except Exception as e:
        st.error(f"Vector database could not be initialized: {e}")
        return

    # Initialize Workflow
    workflow = LegalAssistantWorkflow(vector_db)

    # Query Input
    query = st.text_area(
        "Enter your legal query:", 
        height=100,
        placeholder="Ask about a legal issue..."
    )

    if st.button("Get Legal Insights"):
        if not query:
            st.warning("Please enter a legal query")
            return

        try:
            # Process Query
            result = workflow.process_query(query)

            # Display Results
            st.subheader("🔎 Refined Query")
            st.markdown(result['refined_query'])

            st.subheader("📜 Retrieved Legal Documents")
            st.text_area("Legal Text", value=result['retrieved_documents'], height=200)

            st.subheader("📝 Simplified Legal Summary")
            st.markdown(result['simplified_summary'])

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
