"""Chapter 10: a stateful workflow with human approval, using LangGraph.

Workflow:  classify -> draft_reply -> human_review -> send_reply
                                         |-> (rejected) END

The graph PAUSES at human_review using langgraph's interrupt(), and is
resumed later with Command(resume=...). A checkpointer stores the state
between the pause and the resume.

No language model is required: `draft_fn` is injected, so you can plug in
an LLM later. "Sending" only appends to a list: nothing leaves your machine.
"""
from __future__ import annotations

from typing import Callable, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class TicketState(TypedDict, total=False):
    ticket_text: str
    category: Literal["billing", "technical", "other"]
    draft: str
    approved: bool
    sent: bool


def classify(state: TicketState) -> dict:
    text = state["ticket_text"].lower()
    if any(w in text for w in ("invoice", "refund", "charge", "payment")):
        category = "billing"
    elif any(w in text for w in ("error", "crash", "bug", "login")):
        category = "technical"
    else:
        category = "other"
    return {"category": category}


def make_graph(draft_fn: Callable[[str, str], str], outbox: list[str]):
    """Build and compile the workflow. `outbox` records 'sent' replies."""

    def draft_reply(state: TicketState) -> dict:
        return {"draft": draft_fn(state["category"], state["ticket_text"])}

    def human_review(state: TicketState) -> dict:
        # Execution stops here until someone resumes the graph with a decision.
        decision = interrupt({"question": "Send this reply?", "draft": state["draft"]})
        return {"approved": bool(decision)}

    def send_reply(state: TicketState) -> dict:
        outbox.append(state["draft"])  # simulated side effect
        return {"sent": True}

    def route_after_review(state: TicketState) -> str:
        return "send_reply" if state.get("approved") else END

    builder = StateGraph(TicketState)
    builder.add_node("classify", classify)
    builder.add_node("draft_reply", draft_reply)
    builder.add_node("human_review", human_review)
    builder.add_node("send_reply", send_reply)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "draft_reply")
    builder.add_edge("draft_reply", "human_review")
    builder.add_conditional_edges("human_review", route_after_review,
                                  {"send_reply": "send_reply", END: END})
    builder.add_edge("send_reply", END)
    return builder.compile(checkpointer=InMemorySaver())


def template_drafter(category: str, ticket_text: str) -> str:
    return f"Hello, thanks for contacting us about a {category} issue. We are looking into it."


if __name__ == "__main__":
    outbox: list[str] = []
    graph = make_graph(template_drafter, outbox)
    config = {"configurable": {"thread_id": "ticket-001"}}

    result = graph.invoke({"ticket_text": "I was charged twice on my invoice"}, config)
    print("Paused with:", result["__interrupt__"][0].value)

    answer = input("Approve? [y/N] ").strip().lower() == "y"
    graph.invoke(Command(resume=answer), config)
    print("Outbox:", outbox)
