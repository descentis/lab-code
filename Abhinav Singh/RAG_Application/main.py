import streamlit as st
import bs4
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
import json
import os
import uuid

# Page config
st.set_page_config(page_title="RAG Application ChatBot")

# Chat Title
st.title("RAG APPLICATION CHATBOT")

def fetch_api_keys():
    """Load API keys from environment variable if available else fetch keys from JSON file."""
    if os.getenv('GROQ_API_KEY') and os.getenv('NOMIC_API_KEY'):
        os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
        os.environ['NOMIC_API_KEY'] = os.getenv('NOMIC_API_KEY')

    else:

        with open("api_key.json", "r") as f:
            api_keys = json.load(f)
            os.environ['GROQ_API_KEY'] = api_keys.get('GROQ_API_KEY')
            os.environ['NOMIC_API_KEY'] = api_keys.get('NOMIC_API_KEY')


# Load API keys
fetch_api_keys()

# Initialize Browse_file variable
Browse_file = st.file_uploader("Browse your PDF document", type="pdf")

# Initialize session id and state to keep track of chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chain" not in st.session_state:
    st.session_state.chain = None


def initialize_chain():
    """Initialize the chain."""
    # Initialize LLM
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.3, max_tokens=2000)

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(Browse_file.getvalue())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()  # extracts text into a list of document objects

    # Split text and create vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)  # list of text chunks

    # Use Nomic embeddings
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    # Create in-memory vector store
    st.session_state.vector_store = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)

    # Create retriever
    retriever = st.session_state.vector_store.as_retriever()

    # Set up prompts and chains
    contextualize_q_system_prompt = """
        Given a chat history and the latest user question
        which might refer context in the chat history,
        formulate a standalone question which is relevant and self understandable
        without the chat history, Do NOT answer the question,
        just reformulate it if needed and otherwise return it as it is
        """

    # Chat prompt template for context
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create retriever that utilizes (llm) and a document retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create QA chain
    system_prompt = """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you do not know the answer, say that you don't know.
        Use 5 to 10 sentences maximum to keep the answer concise.

        {context}
        """



    # Chat prompt template for qa
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create chat history
    store = {}

    # Retrieve or initialize ChatMessageHistory
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # to manage chat history and interactions
    st.session_state.chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Remove temporary file
    os.remove("temp.pdf")


# Initialize chain if a file is uploaded
if Browse_file and st.session_state.chain is None:
    initialize_chain()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here "):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(
                {"input": prompt, "pdf_name": Browse_file.name},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            answer = response["answer"]

            st.write(answer)

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
