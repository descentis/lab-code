# beautiful suite to scrap data from webpage
import bs4

# vectorstore - to store scrapped data in form of embeddings
from langchain_community.vectorstores import InMemoryVectorStore

# document_loaders - to load/scrap web based document/data and store in vectorstore
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.document_loaders import PyPDFLoader
# from PyPDF2 import PdfReader

# chat_history - to remember messages
from langchain_core.chat_history import BaseChatMessageHistory
# create history in retriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
#chat message history
from langchain_community.chat_message_histories import ChatMessageHistory

# create template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# stuff chain - combine documents
from langchain.chains.combine_documents import create_stuff_documents_chain

# Runnable - to respond to query in dynamic way
from langchain_core.runnables.history import RunnableWithMessageHistory

# splitter - to split by characters/words/sentences
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChatGroq - use APIs
from langchain_groq import ChatGroq
# embedding
from langchain_nomic import NomicEmbeddings

# ollama embedding
from langchain_community.embeddings import OllamaEmbeddings

# store API key and read in main file
import json

# store APIs
import os

# read API keys file
with open("api_key.json", "r") as f:
   api_key = json.load(f)

# store APIs in virtual environment
os.environ["GROQ_API_KEY"] = api_key["GROQ_API_KEY"]
os.environ["NOMIC_API_KEY"] = api_key["NOMIC_API_KEY"]

# load to model
llm = ChatGroq(model = "llama3-8b-8192")


# create retriever
loader = WebBaseLoader(    
   # web path    
   web_path = ("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"),
   #parse using bs    
   bs_kwargs = dict(     
      parse_only = bs4.SoupStrainer(            
         # check content in division : post-title/post-single/post-header/post-body/post-content/post-footer            
            class_ = ("post-content", "post-title", "post-header")        
            )    
      )
   )

'''
# load pdf
loader = PyPDFLoader('Animal.pdf')
# load content 
pdf = loader.load()
#for doc in pdf:
 #  print(doc.page_content)
'''

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=NomicEmbeddings(model="nomic-embed-text-v1.5"))
# vectorstore = InMemoryVectorStore.from_documents(documents=splits,
#             embedding=OllamaEmbeddings(model="nomic-embed-text:latest"))
retriever = vectorstore.as_retriever()


# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might refer context in the chat history,"
    "formulate a standalone question which is relevant and self understandable"
    "without the chat history, Do NOT answer the question,"
    "just reformulate it if needed and otherwise return it as it is"
)

# template
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
       # placeholder to store history
        MessagesPlaceholder("chat_history"),
        ('human', '{input}')
    ]
)

# retrieve data according to chat history & prompt
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# prompt

system_prompt = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer the question"
    "If you do not know the answer, say that you don't know."
    "Use 5 to 10 sentences maximum to keep the answer concise."
    "\n\n"
    # retrieved from retriever
    "{context}"
)

# question to llm
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder("chat_history"),
        ('human', "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# take data from question asked/ history and create prompt
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
   # check session id
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# rag chain with history
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
            'configurable': {'session_id': "1221"}
        },
    )
    print(response['answer'])