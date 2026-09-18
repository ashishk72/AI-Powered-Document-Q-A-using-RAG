import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from transformers import pipeline


# ---------------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------------

st.set_page_config(page_title="RAG Document QA")

st.title("📄 RAG Document Question Answering")


# ---------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)


# ---------------------------------------------------------
# Process Documents
# ---------------------------------------------------------

documents = []

if uploaded_files:

    for uploaded_file in uploaded_files:

        # Save uploaded PDF temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()

        documents.extend(docs)


    # -----------------------------------------------------
    # Split documents into chunks
    # -----------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(documents)


    # -----------------------------------------------------
    # Create Embeddings
    # -----------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # -----------------------------------------------------
    # Create FAISS Vector Store
    # -----------------------------------------------------

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )


    # -----------------------------------------------------
    # Retriever
    # -----------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


    # -----------------------------------------------------
    # Local Instruction-Following LLM
    # -----------------------------------------------------

    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=200,
        do_sample=False
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )


    # -----------------------------------------------------
    # Grounded RAG Prompt
    # -----------------------------------------------------

    prompt_template = """
You are a document question-answering assistant.

Answer the question ONLY using the provided context.

If the answer cannot be found in the context,
say:

"I could not find this information in the uploaded documents."

Do not use outside knowledge.
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


    # -----------------------------------------------------
    # RAG Chain
    # -----------------------------------------------------

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=True
    )


    # -----------------------------------------------------
    # User Question
    # -----------------------------------------------------

    query = st.text_input(
        "Ask a question from the document",
        key="question"
    )


    if query:

        with st.spinner("Generating answer..."):

            result = qa.invoke({
                "query": query
            })


        answer = result["result"]


        # -------------------------------------------------
        # Display Answer
        # -------------------------------------------------

        st.subheader("Answer:")
        st.write(answer)


        # -------------------------------------------------
        # Display Retrieved Chunks
        # -------------------------------------------------

        st.subheader("Top Retrieved Chunks:")

        for doc in result["source_documents"]:

            st.write(doc.page_content)

            st.write("------")