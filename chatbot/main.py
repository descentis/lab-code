import bs4
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
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
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


# Reading the API key file
with open("api_key.json", "r") as f:
    apis = json.load(f)

'''
# Storing the APIs in the virtual env
os.environ['GROQ_API_KEY'] = apis['GROQ_API_KEY']
os.environ['NOMIC_API_KEY'] = apis['NOMIC_API_KEY']
'''
#llm = ChatGroq(model="llama3-8b-8192")
llm = ChatOllama(model="llama3.2:1b")

# A pdf document loader
file_path = 'Generative AI Course.pdf'
loader = PyPDFLoader(file_path)

'''
# Creating a retriever
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header"))
        ),
)
'''

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
splits = text_splitter.split_documents(docs)
#vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=NomicEmbeddings(model="nomic-embed-text-v1.5"))
vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=OllamaEmbeddings(model="nomic-embed-text:latest"))
retriever = vectorstore.as_retriever()


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

while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
        print('Goodbye')
        break
    response = conversational_rag_chain.invoke(
        {'input': user_input},
        config={
            'configurable': {'session_id': "abc123"}
        },
    )
    print(response['answer'])
