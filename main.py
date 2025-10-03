from dotenv import load_dotenv
import streamlit as st
from preprocessor import predict_next_question as pnq
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Title of the app
st.title("Welcome to the English Tense Tutor Chatbot!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": "I want to learn English tenses."})

if "chain_memory" not in st.session_state:
    st.session_state.chain_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate a bot response (replace this with your AI model if needed)
    bot_response = pnq(user_input=user_input, memory = st.session_state.chain_memory)  # Simple echo bot

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response[0]})
    
    # Updating the memory with the latest interaction
    st.session_state.chain_memory.chat_memory.messages.append(HumanMessage(content=user_input))
    st.session_state.chain_memory.chat_memory.messages.append(AIMessage(content=bot_response[0]))


    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response[0])
