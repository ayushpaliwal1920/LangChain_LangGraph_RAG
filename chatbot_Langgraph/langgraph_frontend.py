import os

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

from langgraph_backend import (
    chatbot,
    chat,
    generate_thread_id,
    retrieve_all_threads
)


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["LANGCHAIN_PROJECT"] = "chatbot-langgraph"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_thread(thread_id: str):

    thread_id = str(thread_id)

    if thread_id not in st.session_state.chat_threads:

        st.session_state.chat_threads.append(thread_id)


def reset_chat():

    new_thread_id = generate_thread_id()

    st.session_state.thread_id = new_thread_id

    st.session_state.messages = []

    add_thread(new_thread_id)


def load_conversation(thread_id: str):

    config = {
        "configurable": {
            "thread_id": str(thread_id)
        }
    }

    try:

        state = chatbot.get_state(config)

        return state.values.get("messages", [])

    except Exception as e:

        st.error(f"Could not load conversation: {e}")

        return []


def convert_messages(messages):

    converted = []

    for msg in messages:

        if isinstance(msg, HumanMessage):

            role = "user"

        elif isinstance(msg, AIMessage):

            role = "assistant"

        else:

            continue

        content = msg.content

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    text = item.get("text")

                    if text:
                        text_parts.append(text)

            content = "".join(text_parts)

        else:

            content = str(content)

        converted.append({
            "role": role,
            "content": content
        })

    return converted


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = generate_thread_id()


if "messages" not in st.session_state:

    st.session_state.messages = []


if "chat_threads" not in st.session_state:

    st.session_state.chat_threads = retrieve_all_threads()


add_thread(st.session_state.thread_id)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 AI Chatbot")

st.caption("Powered by LangGraph + Gemini")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💬 LangGraph Chatbot")


# ============================================================
# NEW CHAT
# ============================================================

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


# ============================================================
# CONVERSATIONS
# ============================================================

st.sidebar.header("My Conversations")


for thread_id in st.session_state.chat_threads:

    if st.sidebar.button(
        f"💬 {thread_id}",
        key=f"conversation_{thread_id}",
        use_container_width=True
    ):

        st.session_state.thread_id = thread_id

        messages = load_conversation(thread_id)

        st.session_state.messages = convert_messages(messages)

        st.rerun()


# ============================================================
# CURRENT THREAD
# ============================================================

st.sidebar.divider()

st.sidebar.write("Current conversation:")

st.sidebar.code(
    st.session_state.thread_id
)


# ============================================================
# DISPLAY HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Ask me anything..."
)


# ============================================================
# PROCESS MESSAGE
# ============================================================

if user_message:

    # User message
    with st.chat_message("user"):

        st.markdown(user_message)

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })


    # Assistant
    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        try:

            for chunk in chat(
                user_message,
                st.session_state.thread_id
            ):

                full_response += chunk

                placeholder.markdown(full_response)


            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:

            st.error(f"❌ Error: {e}")