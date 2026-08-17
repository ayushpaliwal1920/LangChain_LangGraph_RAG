import os
import sqlite3
import uuid
import requests

from dotenv import load_dotenv

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()


# ============================================================
# 1. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# 2. TOOLS
# ============================================================

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform basic arithmetic.

    Supported operations:
    add, sub, mul, div
    """

    try:

        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":

            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed"
                }

            result = first_num / second_num

        else:
            return {
                "error": f"Unsupported operation '{operation}'"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:

        return {
            "error": str(e)
        }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a US stock symbol.

    Examples:
    AAPL, TSLA, MSFT, GOOGL
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is not configured"
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol.upper()}"
        f"&apikey={api_key}"
    )

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


tools = [
    search_tool,
    get_stock_price,
    calculator
]


llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 3. STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 4. CHAT NODE
# ============================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# 5. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 6. CHECKPOINTER
# ============================================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(
    conn=conn
)


# ============================================================
# 7. GRAPH
# ============================================================

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)


chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 8. THREAD FUNCTIONS
# ============================================================

def generate_thread_id():

    return str(uuid.uuid4())


def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        config = checkpoint.config

        thread_id = (
            config
            .get("configurable", {})
            .get("thread_id")
        )

        if thread_id:
            all_threads.add(thread_id)

    return list(all_threads)


# ============================================================
# 9. CHAT STREAM FUNCTION
# ============================================================

def chat(user_message: str, thread_id: str):

    config = {
        "configurable": {
            "thread_id": str(thread_id)
        }
    }

    for chunk in chatbot.stream(
        {
            "messages": [
                HumanMessage(
                    content=user_message
                )
            ]
        },
        config=config,
        stream_mode="messages"
    ):

        message = chunk[0]

        # Only stream normal AI text
        if message.type == "AI":

            content = message.content

            if isinstance(content, str):

                yield content