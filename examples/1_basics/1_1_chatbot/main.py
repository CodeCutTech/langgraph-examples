from typing import Annotated
from dotenv import load_dotenv
from os import getenv

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from examples.utils.stream import interact_with_graph

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

load_dotenv()
LLM_MODEL=getenv('LLM_MODEL')
llm = init_chat_model(LLM_MODEL)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

interact_with_graph(graph)