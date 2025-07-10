from pyexpat import model
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
from retriever import create_retriever
from langchain_core.prompts import ChatPromptTemplate

# Load API KEY information
load_dotenv()

# Enter the project name
logging.langsmith("[Project] PDF RAG")

# Create cache directory
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# Folder for file uploads
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("Local Model Based RAG 💬")

# Code to run only once initially
if "messages" not in st.session_state:
    # Create to save conversation history
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # If no file is uploaded
    st.session_state["chain"] = None

# Create sidebar
with st.sidebar:
    # Create reset button
    clear_btn = st.button("Reset Conversation")

    # File upload
    uploaded_file = st.file_uploader("Upload File", type=["pdf"])

    # Model selection menu
    selected_model = st.selectbox("Select LLM", ["HALO++", "SETBOX"], index=0)

    # Confirm button for model selection
    confirm_model_btn = st.button("Confirm Model Selection")


# Print previous conversations
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# Add a new message
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# Cache the file (to handle long processing tasks)
@st.cache_resource(show_spinner="Processing uploaded file...")
def embed_file(file):
    # Save the uploaded file to the cache directory
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return create_retriever(file_path)


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# Create chain
def create_chain(retriever, model_name="HALO++"):
    # Step 6: Create Prompt
    # Create the prompt
    if model_name == "HALO++":
        # Step 6: Create Prompt
        prompt = load_prompt("prompts/pdf-rag-ollama.yaml", encoding="utf-8")

        # Step 7: Create Language Model (LLM)
        llm = ChatOllama(model="Llma3-HMGICS-SETBOX-Q8:latest", temperature=0)

    elif model_name == "SETBOX":
        # Step 6: Create Prompt
        prompt = load_prompt("prompts/pdf-rag-ollama.yaml", encoding="utf-8")

        # Step 7: Create Language Model (LLM)
        # Load the Ollama model
        llm = ChatOllama(model="Llma3-HMGICS-SETBOX-Q8:latest", temperature=0)

    # Step 8: Create Chain
    chain = (
        {"context": retriever | format_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# Without retriever
def create_chain(model_name="HALO++"):
    # Step 6: Create Prompt
    # Create the prompt
    if model_name == "HALO++":
        # Step 6: Create Prompt
        prompt = ChatPromptTemplate.from_messages(
            [("system", "your are HALO++ analysis expert."), ("human", "{input}")]
        )
        # Step 7: Create Language Model (LLM)
        llm = ChatOllama(model="Llma3-HMGICS-SETBOX-Q8:latest", temperature=0)

    elif model_name == "SETBOX":
        # Step 6: Create Prompt
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a setbox analysis expert."), ("human", "{input}")]
        )
        # Step 7: Create Language Model (LLM)
        # Load the Ollama model
        llm = ChatOllama(model="Llma3-HMGICS-SETBOX-Q8:latest", temperature=0)

    # Step 8: Create Chain
    chain = prompt | llm | RunnablePassthrough()
    return chain


# When a file is uploaded
if uploaded_file:
    # Create retriever after file upload (long processing time expected...)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain
elif confirm_model_btn:
    # Create chain without file upload
    chain = create_chain(model_name=selected_model)
    st.session_state["chain"] = chain

# When the reset button is pressed...
if clear_btn:
    st.session_state["messages"] = []

# Print previous conversation history
print_messages()

# User input
user_input = st.chat_input("Ask what you are curious about!")

# Empty area for displaying warning messages
warning_msg = st.empty()

# If user input is received...
if user_input:
    # Create chain
    chain = st.session_state["chain"]

    if chain is not None:
        # User input
        st.chat_message("user").write(user_input)
        # Streaming call
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # Create an empty space (container) to stream tokens
            container = st.empty()
            stream_response = st.session_state["chain"].stream
            
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # Save the conversation history
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # Display a warning message to upload a file
        warning_msg.error("Please upload a file.")

