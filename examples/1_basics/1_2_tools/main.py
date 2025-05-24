from os import getenv
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

from examples.utils.stream import interact_with_graph

load_dotenv()
LLM_MODEL=getenv('LLM_MODEL')
llm = init_chat_model(LLM_MODEL)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    messages=state["messages"]
    message=llm_with_tools.invoke(messages)
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

interact_with_graph(graph)
