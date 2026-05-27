import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

# -----------------------------------
# LOAD ENV VARIABLES
# -----------------------------------

load_dotenv()

# -----------------------------------
# LANGCHAIN IMPORTS
# -----------------------------------

from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import CharacterTextSplitter

from langchain_classic.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Conversational Multi-Document RAG",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🤖 Conversational Multi-Document RAG Assistant")

st.write(
    "Upload multiple PDF documents and chat with AI using semantic retrieval."
)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("📚 RAG Assistant")

st.sidebar.info(
    """
    Features:
    
    ✅ Multi PDF Upload
    
    ✅ Conversational Chat
    
    ✅ Semantic Search
    
    ✅ FAISS Vector Database
    
    ✅ HuggingFace Embeddings
    
    ✅ Groq LLM
    
    ✅ Retrieval QA Pipeline
    
    ✅ Source Tracking
    """
)

# -----------------------------------
# CLEAR CHAT BUTTON
# -----------------------------------

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type="pdf",
    accept_multiple_files=True
)

# -----------------------------------
# PROCESS DOCUMENTS
# -----------------------------------

if uploaded_files:

    st.sidebar.success(
        f"{len(uploaded_files)} PDF documents uploaded"
    )

    documents = []

    all_text = ""

    # -----------------------------------
    # READ ALL PDFs
    # -----------------------------------

    for uploaded_file in uploaded_files:

        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        documents.append({
            "name": uploaded_file.name,
            "text": text
        })

        all_text += text

    # -----------------------------------
    # SHOW DOCUMENT CONTENT
    # -----------------------------------

    with st.expander("📄 Extracted Document Content"):

        st.text_area(
            "Document Text",
            all_text,
            height=300
        )

    # -----------------------------------
    # TEXT SPLITTING
    # -----------------------------------

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )

    texts = []

    metadatas = []

    for doc in documents:

        chunks = splitter.split_text(doc["text"])

        for chunk in chunks:

            texts.append(chunk)

            metadatas.append({
                "source": doc["name"]
            })

    # -----------------------------------
    # EMBEDDINGS
    # -----------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------------
    # VECTOR DATABASE
    # -----------------------------------

    vectorstore = FAISS.from_texts(
        texts,
        embeddings,
        metadatas=metadatas
    )

    # -----------------------------------
    # RETRIEVER
    # -----------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    # -----------------------------------
    # LLM
    # -----------------------------------

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    # -----------------------------------
    # PROMPT TEMPLATE
    # -----------------------------------

    prompt_template = """

    You are a helpful AI assistant.

    Use ONLY the provided context to answer questions.

    If the answer is not available in the documents,
    say:
    "The information is not available in the uploaded documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # -----------------------------------
    # RETRIEVAL QA CHAIN
    # -----------------------------------

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    # -----------------------------------
    # DOCUMENT SUMMARY
    # -----------------------------------

    if st.sidebar.button("📄 Summarize Documents"):

        with st.spinner("Generating Summary..."):

            summary = qa_chain.run(
                "Give a detailed summary of all uploaded documents"
            )

            st.sidebar.write(summary)

    # -----------------------------------
    # CHAT HISTORY
    # -----------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # DISPLAY OLD MESSAGES

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # -----------------------------------
    # USER INPUT
    # -----------------------------------

    query = st.chat_input(
        "Ask questions about uploaded documents"
    )

    # -----------------------------------
    # PROCESS QUESTION
    # -----------------------------------

    if query:

        # USER MESSAGE

        st.chat_message("user").markdown(query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # -----------------------------------
        # RETRIEVE DOCUMENTS
        # -----------------------------------

        docs = retriever.invoke(query)

        # -----------------------------------
        # SHOW RETRIEVED CHUNKS
        # -----------------------------------

        with st.expander("📄 Retrieved Context"):

            for i, doc in enumerate(docs):

                st.write(f"### Chunk {i+1}")

                st.write(
                    f"📁 Source: {doc.metadata['source']}"
                )

                st.write(doc.page_content)

                st.divider()

        # -----------------------------------
        # GENERATE RESPONSE
        # -----------------------------------

        with st.spinner("Generating AI response..."):

            response = qa_chain.run(query)

        # -----------------------------------
        # SHOW AI RESPONSE
        # -----------------------------------

        with st.chat_message("assistant"):

            st.success(response)

        # -----------------------------------
        # SAVE AI RESPONSE
        # -----------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

    # -----------------------------------
    # DOWNLOAD CHAT
    # -----------------------------------

    chat_history = ""

    for msg in st.session_state.messages:

        role = msg["role"]

        content = msg["content"]

        chat_history += f"{role}: {content}\n\n"

    st.download_button(
        "📥 Download Chat History",
        chat_history,
        file_name="chat_history.txt"
    )

else:

    st.info("Please upload one or more PDF documents.")