import streamlit as st
import os
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Set page config first
st.set_page_config(
    page_title="LightningDoc QA",
    page_icon="⚡",
    layout="centered"
)

# Optimized components
@st.cache_resource
def load_embedding_model():
    """Lightweight embedding model cached for performance"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    """Load optimized CPU-friendly LLM"""
    return Ollama(
        model="tinyllama",  # Microsoft's 3.8B model - faster than TinyLlama
        temperature=0,
        num_thread=4,  # Use 6 CPU threads
        num_ctx=2048,  # Smaller context = faster
        top_k=20,      # Reduce sampling options
        stop=["<|end|>", "<|user|>"]  # Better stopping tokens
    )

# Custom prompt for faster responses
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    <|system|>
    You are a helpful document assistant. Answer the question using ONLY the context below.
    Be concise - maximum 2 sentences. If unsure, say "I couldn't find that in the document."
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
)

# UI Components
st.title("⚡ LightningDoc QA")
st.caption("Upload a PDF and get instant answers - optimized for CPU")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 512, 2048, 1024, help="Smaller chunks = faster processing")
    top_k = st.slider("Retrieval Count", 1, 5, 2, help="Number of document chunks to use")

# File uploader
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# Processing pipeline
if uploaded_file:
    # Step 1: Save and load document
    with st.spinner("Preparing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)  # Delete temp file
    
    # Step 2: Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(pages)
    
    # Step 3: Create vector store
    with st.spinner("Building knowledge base..."):
        embeddings = load_embedding_model()
        vector_db = FAISS.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    
    # Step 4: Initialize QA system
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    
    st.success("Document ready! Ask your question below")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            start_time = time.time()
            result = qa_chain.invoke({"query": prompt})
            response_time = time.time() - start_time
            
            # Stream the response
            for chunk in result["result"].split():
                full_response += chunk + " "
                time.sleep(0.05)  # Simulate streaming
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Show performance metrics
            st.caption(f"⏱️ Response: {response_time:.2f}s | 📚 Sources: {len(result['source_documents'])}")
        
        st.session_state.messages.append({"role": "assistant", "content": result["result"]})