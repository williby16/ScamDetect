from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st

# App title
st.set_page_config(page_title="🕵️‍♀️ ScamDetect")

# Set up the Streamlit framework
st.title('🕵️‍♀️ ScamDetect')
st.caption("🕸️ A Scam Detection chatbot powered by LLAMA3")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Please input your suspicious text."}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Please input your suspicious text."}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# User-provided input
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

# Initialize the Ollama model
llm = Ollama(model="llama3")

# Define a prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a text scam detector. Please respond to the questions"),
        ("user","Question:{question}")
    ]
)

# Create a chain that combines the prompt and the Ollama model
chain = prompt|llm

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get user prompt from the latest user message
            user_prompt = st.session_state.messages[-1]["content"]
            # Invoke the chain with the user prompt
            response = chain.invoke({"question": user_prompt})
            st.write(response)
    # Update chat history
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

    print(st.session_state.messages)
