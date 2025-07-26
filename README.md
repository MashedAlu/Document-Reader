# Document-Reader
Document parser that uses DeepSeek to parse through document and answer simple questions

## Setup Guide

1. Download the app.py, .env and requirements.txt file
2. Run the requirements.txt file to get the libraries needed
3. In the terminal, use 'streamlit run app.py'
4. The app will run on localhost

## Tools and libraries used:

The whole app was coded in Python

Frontend and backend was managed by the python Streamlit library
Langchain was used for the RAG system
As well as PyPDF for the PDF reading
Deepseek API was used for chat
Huggingface for embeddings
FAISS for the vector database

Questions:

1. I used PyPDFLoader extract and split the PDF. While it's effective for plain PDFs, some formatting inconsistencies (like tables or multi-column layouts) can still affect chunk readability.
2. I used a chunk size of 1000 and overlap of 200 using RecursiveCharacterTextSplitter for high overlap and preserving context across chunks.It improves retrieval accuracy during semantic search.
3. I used MiniLM l6 v2 from HuggingFace. I mainly used this because it is lightweight and good for general purpose NLP. It works by finding sentence-level meaning by encoding context and relationships between words into dense vector representations.
4. I used FAISS with the default inner product . It's fast, scalable, and works well with LangChain.
5. I use the same embedding model (all-MiniLM-L6-v2) for both the document chunks and the user query. This keeps their vector representations in the same semantic space, allowing FAISS to retrieve the most relevant chunks. The top-matching chunks are then passed to DeepSeek, which uses them as context to generate an answer
6. In most cases, the results were relevant and good, especially for good queries/prompts but, relevance can drop with vague or overly broad questions.



