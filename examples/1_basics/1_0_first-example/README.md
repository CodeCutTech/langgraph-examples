# Prompting Isn’t Enough: Why LLM Workflows Need Graphs, Not Chains

Ever tried building a support assistant with just prompts or chains—only to watch it hallucinate, forget what it was doing, or get stuck in loops?

Prompt chaining (like LangChain) helps a bit. But real-world workflows don’t move in straight lines. They loop, branch, retry, and rely on shared memory. Prompt chains are recipes; real tasks are flowcharts.

That’s where **LangGraph** comes in.

It treats your logic as a **graph**: each step is a node, transitions are edges, and a shared state evolves as the system runs. It’s like a state machine, but for LLMs—and it’s shockingly useful.

Let’s build a small, concrete example to show you how.

---

## 🔍 Problem: Analyze a Customer’s History

Suppose we want an assistant that reviews a customer’s history:

* Fetch their profile
* Read feedback they’ve left
* Check support tickets
* Summarize it all

Although this is a strict linear pipeline, it might skip steps, retry them, or use early exits. Instead of forcing it into a rigid sequence, we may build a **graph** of steps that share evolving context. On this post, however, we will keep the linear sequence for the educational purpose. 

---

## 🛠️ Setup

To follow along, install the basics:

```bash
pip install langgraph openai pydantic
```

We’ll simulate tools that return structured data using Pydantic. Think of these as stand-ins for real API calls or database queries.

---

## 📦 Types and Tools

First, define the data we'll be working with:

```python
from pydantic import BaseModel

class Profile(BaseModel):
    id: str
    name: str
    segment: str

class Feedback(BaseModel):
    customer_id: str
    comments: list[str]

class Ticket(BaseModel):
    id: str
    status: str
    issue: str
```

Now, our dummy tools. Each one returns part of the customer’s history:

```python
def get_profile(state):
    return {"profile": Profile(id="123", name="Jane", segment="premium")}

def get_feedback(state):
    return {"feedback": Feedback(customer_id="123", comments=["Great product!", "Support was slow."])}

def get_tickets(state):
    return {"tickets": [Ticket(id="t1", status="open", issue="Login failed")]}
```

These operate on a shared `state`, adding structured results at each step.

---

## 🧠 Shared State Model

To clarify what our graph is passing around, define a simple state schema:

```python
class State(BaseModel):
    profile: Profile | None = None
    feedback: Feedback | None = None
    tickets: list[Ticket] | None = None
```

You could enforce this in LangGraph, but even using plain dicts, it's helpful to know what your workflow state is supposed to look like.

---

## 🔗 Define the Graph

Now let’s wire up the workflow using LangGraph:

```python
from langgraph.graph import StateGraph, MessagesState, START, END


builder = StateGraph(dict)  # dict or State, depending on how strict you want to be

builder.add_node("profile", get_profile)
builder.add_node("feedback", get_feedback)
builder.add_node("tickets", get_tickets)

builder.add_edge(START, "profile")
builder.add_edge("profile", "feedback")
builder.add_edge("feedback", "tickets")
builder.add_edge("tickets", END)

graph = builder.compile()
```

Each node gets access to everything upstream. You’re not just passing output to input—you’re growing a shared, persistent state.

---

## ▶️ Run the Workflow

You can stream the graph’s execution step-by-step:

```python
if __name__ == "__main__":
    for step in graph.stream(initial_state, {"recursion_limit": 5}):
        print("--- step ---")
        # Pretty-print the latest system message and structured data
        messages = step.get("messages", [])
        if messages:
            print(f"Last system message: {messages[-1]}")
        # Also print any structured outputs added at this step
        for key in ["profile", "feedback", "tickets"]:
            if key in step:
                print(f"{key.capitalize()} data: {step[key]}")
```

Sample output:

```
--- step ---
Profile data: {'messages': [HumanMessage(content="Provide Jane's tickets", additional_kwargs={}, response_metadata={}, id='9235e1eb-82a2-4f92-b50e-5fb8916f4389'), {'role': 'system', 'content': 'Loaded profile: Jane, segment premium'}]}
--- step ---
Feedback data: {'messages': [HumanMessage(content="Provide Jane's tickets", additional_kwargs={}, response_metadata={}, id='9235e1eb-82a2-4f92-b50e-5fb8916f4389'), SystemMessage(content='Loaded profile: Jane, segment premium', additional_kwargs={}, response_metadata={}, id='0ec6a3f3-8005-4de8-80ca-4781c974d1e3'), SystemMessage(content='Loaded profile: Jane, segment premium', additional_kwargs={}, response_metadata={}, id='02b6b260-173f-4d57-b761-7c348e877fdb'), {'role': 'system', 'content': 'Feedback notes: 2 comments'}]}
--- step ---
Tickets data: {'messages': [HumanMessage(content="Provide Jane's tickets", additional_kwargs={}, response_metadata={}, id='9235e1eb-82a2-4f92-b50e-5fb8916f4389'), SystemMessage(content='Loaded profile: Jane, segment premium', additional_kwargs={}, response_metadata={}, id='0ec6a3f3-8005-4de8-80ca-4781c974d1e3'), SystemMessage(content='Loaded profile: Jane, segment premium', additional_kwargs={}, response_metadata={}, id='02b6b260-173f-4d57-b761-7c348e877fdb'), SystemMessage(content='Feedback notes: 2 comments', additional_kwargs={}, response_metadata={}, id='a063d2fd-0726-4710-acc3-5962683382f2'), SystemMessage(content='Feedback notes: 2 comments', additional_kwargs={}, response_metadata={}, id='635cecd6-1e4d-440c-87b4-bef22812037a'), {'role': 'system', 'content': 'Found 1 open tickets'}]}
```

At each step, you get a full snapshot of the accumulated state—perfect for debugging or introspecting complex flows.

---

## 🔍 Visualize the Flow

LangGraph can even draw your workflow.

In a notebook, try:

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    print("Optional: install graphviz and Mermaid dependencies for rendering.")
```

Output (in Mermaid syntax):

```
graph LR
  __start__ --> profile --> feedback --> tickets --> __end__
```

A clear trail. Linear now — but extensible.

---

## 🧠 Takeaways

* **LangGraph manages memory** across steps—no more juggling prompt injections
* Workflows become **transparent**: inspect state at any point
* **Modular logic**: each tool just mutates shared state
* You can easily **extend** with branches, loops, or conditionals

---

## 🚀 Why This Matters

Prompts are powerful. But orchestration? That’s where most LLM projects break down.

You need something between “just glue some prompts” and “build a full backend.” Graphs give you that middle ground: composable, stateful, and debuggable logic for real-world assistants.

LangGraph is that tool. It's not magic—but it's exactly enough structure to stop duct-taping your AI workflows together.
