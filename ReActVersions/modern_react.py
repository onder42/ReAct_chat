from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def add(a: int, b: int) -> int:
    """This tool adds two numbers together."""
    return a + b

model = ChatOllama(model="qwen2.5:3b")
system_prompt = "You are a precise, tool-using mathematical assistant."

app = create_agent(
    model=model,
    tools=[add]
)

inputs = {"messages": [("user", "add 3 + 4.")]}
for update in app.stream(inputs, stream_mode="updates"):
    print(update)