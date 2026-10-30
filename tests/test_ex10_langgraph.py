from langgraph.types import Command

from ex10_langgraph_workflow import classify, make_graph, template_drafter


def cfg(name):
    return {"configurable": {"thread_id": name}}


def test_classify_categories():
    assert classify({"ticket_text": "Refund my payment"})["category"] == "billing"
    assert classify({"ticket_text": "App crash on login"})["category"] == "technical"
    assert classify({"ticket_text": "Do you have stickers?"})["category"] == "other"


def test_graph_pauses_and_does_not_send_before_approval():
    outbox = []
    graph = make_graph(template_drafter, outbox)
    result = graph.invoke({"ticket_text": "charged twice"}, cfg("a"))
    assert "__interrupt__" in result
    assert outbox == []


def test_graph_sends_after_approval():
    outbox = []
    graph = make_graph(template_drafter, outbox)
    graph.invoke({"ticket_text": "charged twice"}, cfg("b"))
    final = graph.invoke(Command(resume=True), cfg("b"))
    assert final["sent"] is True and len(outbox) == 1


def test_graph_does_not_send_after_rejection():
    outbox = []
    graph = make_graph(template_drafter, outbox)
    graph.invoke({"ticket_text": "charged twice"}, cfg("c"))
    final = graph.invoke(Command(resume=False), cfg("c"))
    assert final.get("sent") is None and outbox == []


def test_threads_are_isolated():
    outbox = []
    graph = make_graph(template_drafter, outbox)
    graph.invoke({"ticket_text": "charged twice"}, cfg("x"))
    graph.invoke({"ticket_text": "app crash"}, cfg("y"))
    graph.invoke(Command(resume=True), cfg("x"))
    assert len(outbox) == 1 and "billing" in outbox[0]
