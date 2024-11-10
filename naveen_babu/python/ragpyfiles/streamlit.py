# First, install required dependencies:
# pip install sentence-transformers langchain-groq transformers torch einops faiss-cpu PyMuPDF streamlit

import os
import json
import streamlit as st
import fitz  # PyMuPDF
import requests
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.docstore.document import Document

# Load API keys from config.json
with open("config.json") as f:
    config = json.load(f)
groq_api_key = config["GROQ_API_KEY"]

# Define function to extract text from PDFs
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text

# Initialize embeddings with trust_remote_code=True
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={'device': 'cpu', 'trust_remote_code': True}
)

# Initialize Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=4096
)

# Initialize Streamlit UI
st.title("Conversational Chatbot with Nomic Embeddings and Groq LLM")

# Step 1: Upload and Process PDFs
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

if uploaded_files:
    # Process documents and create texts
    documents = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        # Split text into smaller chunks (e.g., by paragraphs)
        chunks = text.split('\n\n')
        # Create Document objects
        for chunk in chunks:
            if chunk.strip():  # Only add non-empty chunks
                doc = Document(page_content=chunk, metadata={"source": file.name})
                documents.append(doc)
    
    st.success(f"Documents uploaded successfully! Processed {len(documents)} text chunks.")
    
    # Create vector store using FAISS
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )
    
    # Save the FAISS index
    vectorstore.save_local("faiss_index")
    
    # Initialize conversation memory with output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Create retrieval chain with condense_question_prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True,
        chain_type="stuff"  # Use "stuff" chain type for simpler processing
    )
    
    # Step 2: Chat Interface
    st.write("You can now ask questions based on the uploaded documents.")
    
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    query = st.text_input("Ask a question:")
    if query:
        try:
            # Get response from chain
            result = qa_chain({"question": query})
            answer = result['answer']
            
            # Add to chat history
            st.session_state.chat_history.append(("User", query))
            st.session_state.chat_history.append(("Bot", answer))
            
            # Display chat history
            st.write("Chat History:")
            for role, message in st.session_state.chat_history:
                if role == "User":
                    st.write(f"🧑 {message}")
                else:
                    st.write(f"🤖 {message}")
                
            # Display source documents if available
            if 'source_documents' in result:
                st.write("📚 Sources:")
                for doc in result['source_documents']:
                    st.write(f"- From '{doc.metadata['source']}':")
                    st.write(doc.page_content[:200] + "...")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full error details:", exc_info=True)