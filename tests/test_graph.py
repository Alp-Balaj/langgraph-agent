"""Drive the graph with a scripted fake model: no API key needed."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.types import Command

from langgraph_agent.graph import APPROVE_TOKEN, MAX_REVISIONS, build_graph


def run_until_pause(graph, inp, config):
    graph.invoke(inp, config)
    return graph.get_state(config)


def test_pauses_for_human_then_finishes_on_approval():
    llm = FakeListChatModel(responses=["- a\n- b\n- c", "draft one", APPROVE_TOKEN])
    graph = build_graph(llm)
    config = {"configurable": {"thread_id": "t1"}}

    state = run_until_pause(graph, {"topic": "RAG"}, config)
    assert state.next == ("human_review",)
    assert state.tasks[0].interrupts[0].value["plan"] == "- a\n- b\n- c"

    result = graph.invoke(Command(resume={"approved": True}), config)
    assert result["draft"] == "draft one"
    assert result["revisions"] == 0
    assert result["log"] == ["plan", "human_review:approved", "write", "critique"]
    assert graph.get_state(config).next == ()


def test_rejection_loops_back_to_plan_with_feedback():
    llm = FakeListChatModel(responses=["outline v1", "outline v2", "draft", APPROVE_TOKEN])
    graph = build_graph(llm)
    config = {"configurable": {"thread_id": "t2"}}

    run_until_pause(graph, {"topic": "RAG"}, config)
    graph.invoke(Command(resume={"approved": False, "feedback": "add costs"}), config)
    state = graph.get_state(config)
    assert state.next == ("human_review",)  # paused again on the new plan
    assert state.values["plan"] == "outline v2"
    assert state.values["feedback"] == "add costs"

    result = graph.invoke(Command(resume={"approved": True}), config)
    assert result["log"] == [
        "plan", "human_review:rejected", "plan", "human_review:approved", "write", "critique",
    ]


def test_critique_loop_stops_after_max_revisions():
    responses = ["outline"] + ["draft", "- fix this"] * (MAX_REVISIONS + 1)
    graph = build_graph(FakeListChatModel(responses=responses))
    config = {"configurable": {"thread_id": "t3"}}

    run_until_pause(graph, {"topic": "RAG"}, config)
    result = graph.invoke(Command(resume={"approved": True}), config)
    assert result["revisions"] == MAX_REVISIONS
    assert result["log"].count("write") == MAX_REVISIONS + 1


def test_threads_are_isolated():
    graph = build_graph(FakeListChatModel(responses=["p1", "p2"]))
    run_until_pause(graph, {"topic": "one"}, {"configurable": {"thread_id": "a"}})
    run_until_pause(graph, {"topic": "two"}, {"configurable": {"thread_id": "b"}})
    assert graph.get_state({"configurable": {"thread_id": "a"}}).values["topic"] == "one"
    assert graph.get_state({"configurable": {"thread_id": "b"}}).values["topic"] == "two"


def test_graph_shape():
    graph = build_graph(FakeListChatModel(responses=["x"]))
    mermaid = graph.get_graph().draw_mermaid()
    for node in ["plan", "human_review", "write", "critique"]:
        assert node in mermaid
