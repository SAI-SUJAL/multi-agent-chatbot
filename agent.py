import os
import streamlit as st
import traceback

# Updated Phi Imports
from phi.agent import Agent
from phi.llm.groq import Groq  # Updated import

# Updated Langchain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Groq Imports
from groq import Groq as GroqClient  # Direct Groq client import

# Configuration Constants
DEFAULT_PDF_DIR = "backend/data"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mixtral-8x7b-32768"

# Vector Database Initialization
def initialize_vector_db(pdf_paths, embedding_model_name):
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
        model_name=embedding_model_name
    )

    # Create FAISS Vector Store
    return FAISS.from_documents(texts, embedding_model)

# Utility Functions for Agents
def retrieve_documents(query, vector_db, top_k=5):
    """Retrieve documents from vector database"""
    if vector_db is None:
        st.error("Vector database not initialized.")
        return "Vector database not initialized."
    
    try:
        docs = vector_db.similarity_search(query, k=top_k)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return f"Error retrieving documents: {e}"

def simplify_legal_text(text, groq_client, model):
    """Simplify legal text using LLM"""
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a legal expert who simplifies complex legal text for a non-lawyer audience."},
                {"role": "user", "content": f"Simplify this legal text:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error simplifying text: {e}")
        return f"Error simplifying text: {e}"

# Multi-Agent Workflow Coordinator
class LegalAssistantWorkflow:
    def __init__(self, vector_db, groq_client, llm_model):
        self.vector_db = vector_db
        self.groq_client = groq_client
        self.llm_model = llm_model
        
    def process_query(self, query, top_k=5):
        # Step 1: Refine Query
        try:
            refined_query_response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at refining and clarifying legal queries."},
                    {"role": "user", "content": f"Rephrase and clarify this legal query to ensure precise understanding:\n\n{query}"}
                ]
            )
            refined_query = refined_query_response.choices[0].message.content
        
            # Step 2: Retrieve Documents
            retrieved_docs = retrieve_documents(refined_query, self.vector_db, top_k)
            
            # Step 3: Summarize Retrieved Documents
            summarized_content = simplify_legal_text(retrieved_docs, self.groq_client, self.llm_model)
            
            return {
                "refined_query": refined_query,
                "retrieved_documents": retrieved_docs,
                "simplified_summary": summarized_content
            }
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return {
                "refined_query": "Error in query processing",
                "retrieved_documents": f"Error: {str(e)}",
                "simplified_summary": f"An error occurred: {str(e)}"
            }

# Streamlit Interface
def main():
    st.title("AI Legal Assistant")
    st.subheader("Advanced Legal Information Retrieval")

    # Advanced Settings Expander
    with st.expander("🔧 Advanced Settings"):
        # API Configuration
        st.subheader("API Configuration")
        groq_api_key = st.text_input("Groq API Key", type="password", 
                                     help="Your API key for accessing Groq services")
        
        # PDF Configuration
        st.subheader("PDF Document Configuration")
        pdf_dir = st.text_input("PDF Directory", 
                                value=DEFAULT_PDF_DIR, 
                                help="Directory containing PDF documents to search")
        
        # Dynamically find PDF files
        pdf_files = []
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_pdf_dir = os.path.join(base_dir, pdf_dir)
            pdf_files = [f for f in os.listdir(full_pdf_dir) if f.endswith('.pdf')]
        except Exception as e:
            st.error(f"Error finding PDF files: {e}")
        
        selected_pdfs = st.multiselect(
            "Select PDF Documents", 
            pdf_files, 
            help="Choose which PDF documents to use in the search"
        )
        
        # Embedding Model Configuration
        st.subheader("Embedding Model")
        embedding_model = st.selectbox(
            "Embedding Model", 
            [
                "sentence-transformers/all-MiniLM-L6-v2", 
                "sentence-transformers/all-mpnet-base-v2"
            ],
            help="Select the embedding model for document search"
        )
        
        # LLM Configuration
        st.subheader("Language Model")
        llm_model = st.selectbox(
            "LLM Model", 
            [
                "mixtral-8x7b-32768", 
                "llama2-70b-4096",
                "gemma-7b-it"
            ],
            help="Select the language model for query processing"
        )
        
        # Advanced Search Parameters
        st.subheader("Search Parameters")
        top_k = st.slider(
            "Number of Documents to Retrieve", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Number of most similar documents to retrieve"
        )

    # Proceed Button
    if st.button("Initialize Legal Assistant"):
        # Validate inputs
        if not groq_api_key:
            st.warning("Please enter a Groq API Key")
            return
        
        if not selected_pdfs:
            st.warning("Please select at least one PDF document")
            return

        # Construct full PDF paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_pdf_dir = os.path.join(base_dir, pdf_dir)
        pdf_paths = [os.path.join(full_pdf_dir, pdf) for pdf in selected_pdfs]

        # Initialize Groq Client
        try:
            groq_client = GroqClient(api_key=groq_api_key)
        except Exception as e:
            st.error(f"Error initializing Groq client: {e}")
            return

        # Initialize Vector Database
        try:
            vector_db = initialize_vector_db(pdf_paths, embedding_model)
        except Exception as e:
            st.error(f"Vector database could not be initialized: {e}")
            return

        # Create Workflow
        workflow = LegalAssistantWorkflow(vector_db, groq_client, llm_model)

        # Store workflow in session state for query processing
        st.session_state.workflow = workflow
        st.session_state.top_k = top_k
        st.success("Legal Assistant Initialized Successfully!")

    # Query Processing Section
    if 'workflow' in st.session_state:
        st.subheader("Legal Query")
        query = st.text_area(
            "Enter your legal query:", 
            height=100,
            placeholder="Describe your legal question or concern..."
        )

        if st.button("Get Legal Insights"):
            if not query:
                st.warning("Please enter a legal query")
                return

            try:
                # Process Query
                result = st.session_state.workflow.process_query(
                    query, 
                    top_k=st.session_state.top_k
                )

                # Display Results
                st.subheader("Query Analysis")
                st.markdown(f"**Refined Query:** {result['refined_query']}")

                st.subheader("Retrieved Legal Documents")
                st.text_area("Retrieved Documents", 
                             value=result['retrieved_documents'], 
                             height=200)

                st.subheader("Simplified Legal Summary")
                st.markdown(result['simplified_summary'])

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
