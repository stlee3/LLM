from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# API key and project settings
load_dotenv()
logging.langsmith("CSV Agent chatbot")

# Streamlit app settings
st.title("CSV Data Analysis Chatbot 💬")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize list to store conversation messages


# Define constants
class MessageRole:
    """
    Class to define message roles.
    """

    USER = "user"  # User message role
    ASSISTANT = "assistant"  # Assistant message role


class MessageType:
    """
    Class to define message types.
    """

    TEXT = "text"  # Text message
    FIGURE = "figure"  # Figure message
    CODE = "code"  # Code message
    DATAFRAME = "dataframe"  # DataFrame message


# Message related functions
def print_messages():
    """
    Function to display stored messages on the screen.
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)  # Display text message
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)  # Display figure message
                    elif message_type == MessageType.CODE:
                        with st.status("Code Output", expanded=False):
                            st.code(
                                message_content, language="python"
                            )  # Display code message
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)  # Display DataFrame message
                else:
                    raise ValueError(f"Unknown content type: {content}")


def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    Function to store a new message.

    Args:
        role (MessageRole): Message role (user or assistant)
        content (List[Union[MessageType, str]]): Message content
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend(
            [content]
        )  # Merge consecutive messages with the same role
    else:
        messages.append([role, [content]])  # Add new message with a new role


# Sidebar settings
with st.sidebar:
    clear_btn = st.button("Reset Conversation")  # Button to reset the conversation
    uploaded_file = st.file_uploader(
        "Please upload a CSV file.", type=["csv"], accept_multiple_files=True
    )  # CSV file upload feature
    selected_model = st.selectbox(
        "Please select an OpenAI model.", ["gpt-3.5-turbo", "gpt-3.5-turbo"], index=0
    )  # OpenAI model selection option
    apply_btn = st.button("Start Data Analysis")  # Button to start data analysis


# Callback functions
def tool_callback(tool) -> None:
    """
    Callback function to process the result of a tool execution.

    Args:
        tool (dict): Information about the executed tool
    """
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query:
                df_in_result = None
                with st.status("Analyzing data...", expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in st.session_state:
                        result = st.session_state["python_tool"].invoke(
                            {"query": query}
                        )
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(label="Code Output", state="complete", expanded=False)

                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_message(
                        MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result]
                    )

                if "plt.show" in query:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])

                return result
            else:
                st.error("DataFrame is not defined. Please upload a CSV file first.")
                return


def observation_callback(observation) -> None:
    """
    Callback function to process the observation result.

    Args:
        observation (dict): Observation result
    """
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages"][-1][
                1
            ].clear()  # Delete the last message if an error occurs


def result_callback(result: str) -> None:
    """
    Callback function to process the final result.

    Args:
        result (str): Final result
    """
    pass  # Currently does nothing


# agent 
def create_agent(dataframe, selected_model="gpt-3.5-turbo"):
    """
    Function to create a DataFrame agent.

    Args:
        dataframe (pd.DataFrame): DataFrame to analyze
        selected_model (str, optional): OpenAI model to use. Default is "gpt-3.5-turbo"

    Returns:
        Agent: Created DataFrame agent
    """
    return create_pandas_dataframe_agent(
        ChatOpenAI(model=selected_model, temperature=0),
        dataframe,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix="You are a professional data analyst and expert in Pandas. "
        "You must use Pandas DataFrame(`df`) to answer user's request. "
        "If you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
        "I prefer seaborn code for visualization, but you can use matplotlib as well."
        "\n\n<Visualization Preference>\n"
        "- [IMPORTANT] Use `English` for your visualization title and labels."
        "- `muted` cmap, white background, and no grid for your visualization."
        "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable. "
        "The language of final answer should be written in english. "
        "\n\n###\n\n<Column Guidelines>\n"
        "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n",
    )


# Question handling function
def ask(query):
    """
    Function to handle user questions and generate responses.

    Args:
        query (str): User question
    """
    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent"]
        response = agent.stream({"input": query})

        ai_answer = ""
        parser_callback = AgentCallbacks(
            tool_callback, observation_callback, result_callback
        )
        stream_parser = AgentStreamParser(parser_callback)

        with st.chat_message("assistant"):
            for step in response:
                stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]
            st.write(ai_answer)

        add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


# Main logic
if clear_btn:
    st.session_state["messages"] = []  # Clear conversation

if apply_btn and uploaded_file:
    loaded_data = pd.read_csv(uploaded_file)  # Load CSV file
    st.session_state["df"] = loaded_data  # Save DataFrame
    st.session_state["python_tool"] = (
        PythonAstREPLTool()
    )  # Create Python execution tool
    st.session_state["python_tool"].locals[
        "df"
    ] = loaded_data  # Add DataFrame to Python execution environment
    st.session_state["agent"] = create_agent(
        loaded_data, selected_model
    )  # Create agent
    st.success("Setup complete. Please start the conversation!")
elif apply_btn:
    st.warning("Please upload a file.")

print_messages()  # Display stored messages

user_input = st.chat_input("Ask anything you are curious about!")  # Get user input
if user_input:
    ask(user_input)  # Handle user question
