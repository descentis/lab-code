import json
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("Orion Chat Bot")

# Reading the API key file
with open("C:\\Users\\harshan.k\\PycharmProjects\\Orionchatbot\\.venv\\apikey.json", "r") as f:
    apis = json.load(f)

# Storing the APIs in the virtual env
os.environ['GROQ_API_KEY'] = apis['GROQ_API_KEY']


# PDF File loader
def file_loader(filename):
    loader = PyPDFLoader(filename)
    page_contents = loader.load()
    print("File Loaded Successfully ")
    print("Creating the embeddings:::")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(page_contents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = InMemoryVectorStore.from_documents(documents=splits,
                                                     embedding=embeddings)

    print("Embeddings Created:::::::::")
    return vectorstore


llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.4,
    max_tokens=300,
)

retriever = file_loader("Generative AI Course.pdf").as_retriever()

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might refer context in the chat history,"
    "formulate a standalone question which is relevant and self understandable"
    "without the chat history, Do NOT answer the question,"
    "just reformulate it if needed and otherwise return it as it is"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ('human', '{input}')
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answering the question

system_prompt = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer the question"
    "If you do not know the answer, say that you don't know."
    "Use 5 to 10 sentences maximum to keep the answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder("chat_history"),
        ('human', "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Create a chat history
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.text_input("Say Something...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = conversational_rag_chain.invoke(
            {'input': user_input},
            config={
                'configurable': {'session_id': "abc123"}
            },
        )

        assistant_response = response["answer"]
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})