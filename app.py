from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader

# Create a Streamlit app
st.title("AI-Powered Document Q&A")

# Load document to streamlit
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
# Add this to your Streamlit app instead of importing
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


# If a file is uploaded, create the TextSplitter and vector database
if uploaded_file :

    # Code to work around document loader from Streamlit and make it readable by langchain
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name

    
    # Load document and split it into chunks for efficient retrieval.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = load_document(temp_file)
    chunks = text_splitter.split_documents(chunks)


    # Message user that document is being processed with time emoji
    st.write("Processing document... :watch:")
    load_dotenv()
# Generate embeddings
    # Embeddings are numerical vector representations of data, typically used to capture relationships, similarities,
    # and meanings in a way that machines can understand. They are widely used in Natural Language Processing (NLP),
    # recommender systems, and search engines.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Can also use HuggingFaceEmbeddings
    # from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector database containing chunks and embeddings
    vector_db = FAISS.from_documents(chunks, embeddings)
    # Create a document retriever
    retriever = vector_db.as_retriever()
    
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
    llm = ChatOpenAI(api_key=DEEPSEEK_API_KEY,base_url="https://api.deepseek.com/v1",model="deepseek-chat",temperature=0)
    # Create a system prompt
    # It sets the overall context for the model.
    # It influences tone, style, and focus before user interaction starts.
    # Unlike user inputs, a system prompt is not visible to the end user.

    system_prompt = (
        "You are a helpful assistant. Use the given context to answer the question."
        "If you don't know the answer, say you don't know. "
        "{context}"
    )

    # Create a prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create a chain
    # It creates a StuffDocumentsChain, which takes multiple documents (text data) and "stuffs" them together before passing them to the LLM for processing.

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Creates the RAG
    chain = create_retrieval_chain(retriever, question_answer_chain)
    # Streamlit input for question
    question = st.text_input("Ask a question about the document:")
    if question:
        response = chain.invoke({"input": question})['answer']
        st.write(response)
