import streamlit as st
from MainCode import *


#Adjusted the below code to work from GUI
# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
#         print('Goodbye')
#         break
#     response = conversational_rag_chain.invoke(
#         {'input': user_input},
#         config={
#             'configurable': {'session_id': "abc123"}
#         },
#     )
#     print(response['answer'])


st.title("Orion Innovation for U! ")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

chat_display = st.empty()
user_input = st.text_input("You: ", key="userInput")

# Process input when the user submits
if st.button("Send"):
    if user_input:
        print("User input: ", user_input)
        response = conversational_rag_chain.invoke(
            {'input': user_input},
            config={
                'configurable': {'session_id': "abc123"}
            },
        )
        # Store the conversation history
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"AI: {response['answer']}")

        # Update chat display
        chat_display.text("\n".join(st.session_state.chat_history))