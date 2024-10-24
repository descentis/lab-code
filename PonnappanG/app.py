from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import json
import os
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

#with open("api_key.json", "r") as f:
#	apis = json.load(f)

os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['NOMIC_API_KEY'] = st.secrets['NOMIC_API_KEY']

def load_document(file_path):
	loader = PyPDFLoader(file_path)
	documents = loader.load()
	return documents

def create_vectorstore(documents):
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
	splits = text_splitter.split_documents(documents)
	vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=NomicEmbeddings(model="nomic-embed-text-v1.5"))
	return vectorstore

def create_chain(vectorstore):
	llm= ChatGroq(
		model= 'llama3-8b-8192',
		temperature = 0.6
	)
	retriever = vectorstore.as_retriever()
	memory = ConversationBufferMemory(
		llm=llm,
		output_key="answer",
		memory_key="chat_history",
		return_messages = True
	)
	chain = ConversationalRetrievalChain.from_llm(
		llm=llm,
		retriever=retriever,
		chain_type="map_reduce",
		memory=memory,
		verbose=True
	)
	return chain
#Streamlit app built
st.set_page_config(
	page_title="A basic Chatbot",
	layout='centered'
)
st.title("RAG with Streamlit")

if 'chat_history' not in st.session_state:
	st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload your PDF file here", type=["pdf"])

if uploaded_file:
	file_path = f"{working_dir}/{uploaded_file.name}"
	with open(file_path, "wb") as f:
		f.write(uploaded_file.getbuffer())

	if "vectorstore" not in st.session_state:
		st.session_state.vectorstore = create_vectorstore(load_document(file_path))
	if "conversation_chain" not in st.session_state:
		st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
	with st.chat_message(message['role']):
		st.markdown(message['content'])

user_input = st.chat_input("Talk to the bot..")

if user_input:
	st.session_state.chat_history.append({'role': "user", "content": user_input})
	with st.chat_message("user"):
		st.markdown(user_input)

	with st.chat_message("assistant"):
		response = st.session_state.conversation_chain({'question': user_input})
		assistant_response = response["answer"]
		st.markdown(assistant_response)
	st.session_state.chat_history.append({'role': "assistant", "content": assistant_response})


