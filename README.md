# **📜 Multi-Agent Legal Chatbot**  

🚀 This project is a **multi-agent AI chatbot** designed to retrieve legal information, simplify complex legal concepts, and deliver clear, concise answers.  

It leverages:  
✅ **FAISS** – Fast vector-based search for retrieving relevant legal content.  
✅ **OpenAI GPT** – Summarizes legal text into user-friendly explanations.  
✅ **FastAPI** – Provides a robust and scalable backend for handling queries.  
✅ **MiniLM Embeddings** – Converts legal documents into vectorized representations for efficient retrieval.  
✅ **Streamlit Frontend** – Interactive UI for querying legal information with AI-powered insights.  

---

## **📌 Setup & Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/multi-agent-chatbot.git
cd multi-agent-chatbot
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Configure API Keys**  
Before running the chatbot, **add your OpenAI API key** in the `.env` file inside the `backend/` directory:  

```bash
OPENAI_API_KEY=your-api-key-here
```

---

## **🚀 Running the Application**  

### **1️⃣ Prepare FAISS Index (Required for Searching)**
If you have modified **searching methods** (L2, COSINE, HNSW) or wish to increase the number of retrieved matches (`k` in `query_agent()`), regenerate the FAISS index by running:

```bash
cd backend
python main.py
```

This will **process legal documents** and create an updated `faiss_index.bin`.

---

### **2️⃣ Start the Backend (FastAPI)**
Run the following command inside the `backend/` directory:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### **3️⃣ Start the Frontend (Streamlit)**
Run the following command inside the `frontend/` directory:

```bash
streamlit run app.py
```

---

## **📂 Directory Structure**
```
📦 multi-agent-chatbot
 ┣ 📂 backend                 # FastAPI backend
 ┃ ┣ 📂 data                  # Folder for legal PDFs
 ┃ ┣ 📜 main.py               # Backend application logic
 ┃ ┣ 📜 faiss_index.bin       # FAISS index (vector database)
 ┃ ┣ 📜 text_mappings.pkl     # Stores document-text mappings
 ┃ ┣ 📜 requirements.txt      # Dependencies for backend
 ┃ ┣ 📜 .env.example          # Example environment file for API keys
 ┣ 📂 frontend                # Streamlit frontend
 ┃ ┣ 📜 app.py                # Chatbot UI
 ┣ 📜 README.md               # Project description
```

---

## **📝 How It Works**
1️⃣ The **Query Agent** retrieves relevant sections from legal documents based on user input.  
2️⃣ The **Summarization Agent** converts complex legal jargon into **plain language** explanations.  
3️⃣ FAISS **quickly searches through embedded legal text** to find the best matches.  
4️⃣ OpenAI GPT **refines the response** to ensure clarity and accuracy.  

📌 **Example Query:**  
User: *"What are the steps to file a lawsuit in India?"*  

💡 **Bot Response:**  
```txt
Filing a lawsuit in India involves:
1. Preparing legal documents  
2. Submitting a petition in court  
3. Serving a notice to the opposing party  
4. Attending hearings and following court procedures  
Would you like more details on any step?
```

---

## **🛠️ Troubleshooting**
🔹 **FAISS not retrieving results?**  
   - Ensure that `faiss_index.bin` exists and is **not empty** (`Index Size > 0`).  
   - If needed, **delete and rebuild FAISS**:  
     ```bash
     rm backend/faiss_index.bin backend/text_mappings.pkl
     python backend/main.py
     ```

🔹 **Error: No API key provided?**  
   - Make sure the OpenAI API key is set in `.env` and loaded in `main.py`:
     ```python
     from dotenv import load_dotenv
     load_dotenv()
     ```

🔹 **Need more relevant results?**  
   - Increase `top_k` in `query_agent()` inside `main.py`:
     ```python
     def query_agent(user_query, top_k=10):
     ```

---

## **📜 License**
MIT License - Free to use, modify, and distribute.

---

### 🎯 **Contributions & Feedback**
🔹 Open to collaborations! Feel free to submit **issues** or **pull requests**.  
🔹 **Contact:** Your Name - [GitHub Profile](https://github.com/yourusername)  

🚀 **Start Exploring Legal Insights with AI Today!** 🏛️
