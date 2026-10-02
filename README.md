# 💬 LangGraph Chatbot on Groq — A Streaming Conversational Loop in One Graph Node (Agentic AI #8)

A minimal, fast terminal chatbot built as a **LangGraph state graph** and powered by **Groq's LPU inference** running the open-weights **`openai/gpt-oss-20b`** model. One node, one typed state, streamed responses, and an interactive loop with exit keywords — the smallest complete LangGraph application, and a deliberate switch away from OpenAI to a different, much faster inference provider.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-State%20Graph-1C3C3C)
![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-F55036)
![Model](https://img.shields.io/badge/Model-gpt--oss--20b-412991)
![Agentic AI](https://img.shields.io/badge/Series-Agentic%20AI%20%2308-8A2BE2)

---

## 📌 Overview

The previous LangGraph project in this series built up to a full ReAct tool-calling agent. This one goes the other way: it strips LangGraph down to its **smallest useful shape** — a single `chatbot` node between `START` and `END` — and wraps it in a real interactive chat loop with **streamed output**.

Two things are new compared with the earlier notebooks:

1. **A different LLM provider** — `ChatGroq` instead of `ChatOpenAI`, running on Groq's custom LPU hardware
2. **An open-weights model** — `openai/gpt-oss-20b`, rather than a closed, hosted-only model

Because the graph talks to the model only through LangChain's chat model interface, swapping providers is a two-line change — the graph itself does not change at all.

---

## 🏗️ The Graph

```
START ──► chatbot ──► END
```

```
user input ──► graph.stream({"messages": [user message]})
                        │
                        ▼
              chatbot(state) ──► llm.invoke(state["messages"])
                        │
                        ▼
              {"messages": [AI reply]}  ──► merged into state by add_messages
                        │
                        ▼
              stream event ──► print "Assistant BOT : ..."
```

---

## 🔬 Part 1 — Connecting to Groq

```python
GROQ_API_KEY = getpass("Enter your GROQ API key here :")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-20b")
```

| Choice | Why it matters |
|---|---|
| `getpass()` | The key is typed at runtime and never stored in the notebook or code |
| `ChatGroq` | Groq runs inference on LPUs built for fast token generation, so replies stream quickly |
| `openai/gpt-oss-20b` | An open-weights 20B model — no dependence on one closed model vendor |

---

## 🔬 Part 2 — State and the Single Node

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
```

- **`State`** holds one field, `messages`
- **`add_messages`** is a **reducer**: the node returns only the *new* message, and LangGraph appends it to the existing list instead of overwriting it
- **`chatbot`** is a plain function: read the messages, call the LLM, return the reply

### Building and Compiling the Graph
```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))
```
`draw_mermaid_png()` renders the compiled graph as a diagram — a quick visual check that the wiring is what you meant before running anything.

---

## 🔬 Part 3 — Streaming Responses

```python
def langgraph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if "messages" in value and len(value["messages"]) > 0:
                if value["messages"][-1].type == "ai":
                    print("Assistant BOT :", value["messages"][-1].content)
```

`graph.stream()` yields one event **per node that runs**, instead of waiting for the whole graph to finish like `graph.invoke()`. Each event is a dictionary keyed by node name; the function takes the last message from each update and prints it only if it came from the AI. With more nodes (tools, routers), the same loop would show each step as it happens.

---

## 🔬 Part 4 — The Interactive Chat Loop

```python
while True:
    try:
        user_input = input("Enter your queries here what you want :")
        if user_input.lower() in ["bye", "close", "quit", "terminate", "exit", "see you"]:
            print("Thank you for asking questions, bye and take care!")
            break
        langgraph_updates(user_input)
    except Exception as e:
        print(f"An error occured: {e}")
        break
```

- **Six exit keywords** end the session cleanly
- **`try/except`** stops the loop with a readable message instead of a stack trace if the API call fails (bad key, rate limit, network)

---

## ⚠️ A Limitation Worth Knowing: No Memory Between Turns

Every call to `langgraph_updates()` starts `graph.stream()` with a **fresh state** containing only the newest user message. The graph is compiled **without a checkpointer**, so nothing from earlier turns is kept:

```
You: My name is Kailas.
Bot: Nice to meet you, Kailas!
You: What is my name?
Bot: I don't know your name.   ← previous turn is gone
```

`add_messages` merges messages *within* one run; it does not persist them *across* runs. The fix is LangGraph's built-in persistence:

```python
from langgraph.checkpoint.memory import MemorySaver

graph = graph_builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-1"}}

graph.stream({"messages": [...]}, config)   # same thread_id → remembers the conversation
```

---

## 🗂️ Repository Structure

```
langgraph-chatbot-groq/
├── AgenticAI_07_Langhgraph_Chatbot.ipynb   # Main notebook
├── requirements.txt                          # Dependencies
├── .gitignore                                # Keeps secrets out of git
├── .env.example                              # Template for required environment variables
└── README.md                                 # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A free [Groq API key](https://console.groq.com/keys)

### Installation

```bash
git clone https://github.com/Kailaswadje/langgraph-chatbot-groq.git
cd langgraph-chatbot-groq

pip install -r requirements.txt

jupyter notebook AgenticAI_07_Langhgraph_Chatbot.ipynb
```

> ⚠️ The notebook asks for the key with `getpass()` — keep it that way. Type an exit keyword (`bye`, `exit`, `quit` …) to stop the loop, otherwise the cell keeps waiting for input. Clear notebook outputs before pushing.

---

## 🧠 Key Takeaways

- **The smallest LangGraph app is still a real graph** — `START → chatbot → END` uses the same State/Node/Edge model as a full multi-agent system
- **`add_messages` merges, it does not persist** — it appends within one run; memory across turns needs a checkpointer and a `thread_id`
- **`graph.stream()` beats `graph.invoke()` for chat** — users see output per node instead of waiting for the whole graph
- **Provider portability is real** — moving from OpenAI to Groq changed two lines; the graph, state and loop stayed identical
- **Open-weights models are production-viable** — `gpt-oss-20b` on Groq gives fast, low-cost responses without locking into one closed model

---

## 📚 Agentic AI Series Context

| Part | Project | Framework | Focus |
|---|---|---|---|
| 01 | AutoGen Agent Fundamentals | AutoGen | `ConversableAgent`, peer-to-peer negotiation |
| 02 | UserProxyAgent & Sequential Chat | AutoGen | Human-facing coordination, pipeline handoffs |
| 03 | Group Chat, State Flow & Nested Chat | AutoGen | Multi-agent teams, deterministic orchestration |
| 04 | CrewAI Fundamentals | CrewAI | Structured agents, task/agent separation |
| 05 | CrewAI with Real Web Search | CrewAI | Tool-equipped agents, sequential context |
| 06 | LangGraph Fundamentals | LangGraph | State graphs, conditional routing, ReAct tool loop |
| 07 | AutoGen + RAG for E-Commerce | AutoGen | Agent-based retrieval, async orchestration |
| **08 (this repo)** | LangGraph Chatbot on Groq | **LangGraph + Groq** | Minimal graph, streaming, open-weights model |

---

## 🔮 Possible Extensions

- [ ] Add `MemorySaver` with a `thread_id` so the bot remembers earlier turns
- [ ] Add a `SystemMessage` persona (already imported but not yet used)
- [ ] Attach tools with `ToolNode` + `tools_condition` (also imported) to turn the chatbot into a ReAct agent
- [ ] Stream token by token with `stream_mode="messages"` instead of whole messages
- [ ] Wrap the graph in a Streamlit chat UI

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If this showed you how small a working LangGraph app can be, consider giving it a star!
