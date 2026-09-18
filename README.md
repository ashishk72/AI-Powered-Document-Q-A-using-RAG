# AI-Powered Document Q&A using RAG

An AI-powered document question-answering system that allows users to ask questions about the content of PDF documents. The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents and generate context-aware answers using a Large Language Model (LLM).

##  Features

- Upload and process multiple PDF documents
- Extract text from PDF files
- Split documents into smaller chunks for efficient retrieval
- Generate vector embeddings for document chunks
- Store and search embeddings using FAISS
- Retrieve relevant document content based on user queries
- Generate context-aware answers using an LLM
- Conversational question answering
- Source-based responses to improve answer grounding

##  Architecture

The system follows a Retrieval-Augmented Generation pipeline:

```text
PDF Documents
      ↓
Document Loading
      ↓
Text Extraction
      ↓
Document Chunking
      ↓
Text Embeddings
      ↓
FAISS Vector Database
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
LLM + Prompt
      ↓
Generated Answer

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Document Loading | LangChain PyPDFLoader |
| Text Chunking | RecursiveCharacterTextSplitter |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| Language Model | google/flan-t5-base |
| LLM Framework | LangChain RetrievalQA |
| Language | Python 3.10+ |

---
```
##  Project Structure

```
rag-document-qa/
│
├── app.py                      ← Main Streamlit application
├── requirements.txt            ← All dependencies
├── README.md                   ← Project documentation
├── .gitignore
├── rag_demo_document.pdf       ← Sample RAG demo document
├── ai_ml_dl_demo.pdf           ← AI/ML/DL reference document
├── data_science_nlp_demo.pdf   ← Data Science & NLP document
└── paracetamol.pdf             ← Medical document sample
```

---

##  How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/ashishk72/AI-Powered-Document-Q-A-using-RAG
cd rag-document-qa
```

### Step 2 — Create virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the application
```bash
streamlit run app.py
```

### Step 5 — Open in browser
The app will automatically open at `http://localhost:8501`

---

##  How to Use

1. Click **Browse files** and upload one or more PDF documents
2. Wait for the system to process and index the documents
3. Type your question in the text box
4. The system retrieves relevant chunks and generates an answer
5. Scroll down to see the source document chunks used

---

##  Example Questions to Try

Using the included demo PDFs:

- *"What is Artificial Intelligence?"*
- *"What is Machine Learning?"*
- *"What is Natural Language Processing?"*
- *"What is Deep Learning?"*
- *"What is Retrieval-Augmented Generation?"*
- *"What are the uses of Paracetamol?"*

---

##  How RAG Works — Simple Explanation

Traditional language models answer from memory (training data). RAG is different:

1. **Retrieve** — find the most relevant parts of YOUR documents
2. **Augment** — add that context to the question
3. **Generate** — LLM answers based on YOUR documents, not general knowledge

This means the model answers from your actual documents — reducing hallucination and giving accurate, grounded responses.

---

##  Key Technical Decisions

**Why all-MiniLM-L6-v2?**
Lightweight (80MB), fast inference, strong semantic similarity performance. Ideal for local deployment without GPU.

**Why FAISS?**
Facebook's FAISS library performs approximate nearest neighbour search in milliseconds even on millions of vectors. No external server needed — runs entirely in memory.

**Why chunk_overlap=100?**
Prevents answers from being missed when they span chunk boundaries. The 100-character overlap ensures context continuity between adjacent chunks.

**Why k=3 retrieval?**
Balances context richness with prompt length. Three chunks provide enough context for most questions without overwhelming the LLM's input capacity.

**Why text2text-generation pipeline?**
flan-t5-base is a sequence-to-sequence model designed for text2text-generation tasks. Using this pipeline type ensures the model returns only the generated answer — not the full prompt template.

---

##  Possible Improvements

- [ ] Persist FAISS index to disk to avoid re-indexing on every run
- [ ] Upgrade LLM to `flan-t5-large` or API-based model for better answer quality
- [ ] Add support for `.txt` and `.docx` file formats
- [ ] Add conversation memory for follow-up questions
- [ ] Deploy on Streamlit Community Cloud for public access

---

##  Author

**Ashish Kumar**
M.Tech, Computer Science and Engineering
National Institute of Technology, Rourkela

- LinkedIn: [linkedin.com/in/ashish-kumar-061289227](https://www.linkedin.com/in/ashish-kumar-061289227)
- GitHub: [github.com/ashishk72](https://https://github.com/ashishk72)
- Email: ashishk07376@gmail.com

---


