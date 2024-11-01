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

#Page config
st.set_page_config(page_title="PDF RAG Bot")

# Chat Title
st.title("PDF RAG Chat Bot Using Groq API (model = llama3-8b-8192)")

def load_api_keys():
    """Load API keys from a JSON file or environment variables."""
    try:
        with open("api_key.json", "r") as f:
            api_keys = json.load(f)
            os.environ['GROQ_API_KEY'] = api_keys['GROQ_API_KEY']
            os.environ['NOMIC_API_KEY'] = api_keys['NOMIC_API_KEY']
    except FileNotFoundError:
        st.warning("API keys JSON file not found. Falling back to environment variables from GitHub Secrets.")

    # Retrieve from environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")

    # Check if the keys were retrieved from environment variables
    if not GROQ_API_KEY or not NOMIC_API_KEY:
        st.error("API keys are not set in environment variables. Please check your GitHub Secrets.")
        return None, None

    return GROQ_API_KEY, NOMIC_API_KEY


# Load API keys
GROQ_API_KEY, NOMIC_API_KEY = load_api_keys()

# Initialize session id and state to keep track of chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

def initialize_chain():
    # Initialize LLM
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, max_tokens=2024)

    # Load and process document
    if uploaded_file:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        # Debug - Print loaded documents
        #print("Loaded Documents:", docs)

        # Verify document loading
        if not docs:
            st.error("The PDF could not be loaded.")
            return False

        # Split text and create vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Debug - Print split documents
        #print("Split Documents:", splits)

        # Verify text splitting
        if not splits:
            st.error("The document could not be split into chunks.")
            return False

        # Use Nomic embeddings
        embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

        # Create  in-memory vector store
        st.session_state.vector_store = InMemoryVectorStore.from_documents(documents=splits,embedding=embeddings)

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
        # to include chat history context into the query formulation. The contextualize_q_prompt
        # helps in refining user questions based on previous interactions, improving the relevance of responses.
        history_aware_retriever = create_history_aware_retriever(
            llm,retriever,contextualize_q_prompt
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

        # retrieve or initialize ChatMessageHistory
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
                st.write("Session ID : ", session_id)
                print("Session ID : ", session_id)
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
        return True
    return False


# Initialize chain when file is uploaded
if uploaded_file and not st.session_state.chain:
    with st.spinner("Processing document..."):
        if initialize_chain():
            st.success("Document processed! You can now start chatting.")
        else:
            st.error("Failed to process the document.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document"):
    if not st.session_state.chain:
        st.error("Please upload a document first!")
        st.stop()

    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(
                {"input": prompt, "pdf_name": st.session_state.pdf_name},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            answer = response["answer"]

            st.write(answer)

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
