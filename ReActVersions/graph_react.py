from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



@tool
def add(a: int, b:int) -> int:
    '''this tool adds to numbers'''
    return a + b


tools = [add]
model = ChatOllama(model="qwen2.5:3b").bind_tools(tools=tools)


def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content= "You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

 
def should_contuinue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def save_mermaid(app):
    # Save the mermaid diagram directly as a PNG file
    with open("graph_diagram.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())

    print("Graph diagram saved successfully as 'graph_diagram.png'!")


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node) 

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_contuinue,
    {
        "continue": "tools",
        "end": END,
    }
)

graph.add_edge("tools", "our_agent")
app = graph.compile()


def print_stream(stream):
    for s in stream:
        # values mode streams the entire state dict under the node name key
        for node_name, state in s.items():
            if "messages" in state and state["messages"]:
                message = state["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

inputs = {"messages": [("user", "add 3 + 4.")]}
print_stream(app.stream(inputs, stream_mode="updates"))
save_mermaid(app) 