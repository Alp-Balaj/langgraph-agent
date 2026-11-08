"""A brief-writing agent that shows the four LangGraph basics.

STATE   BriefState: a typed dict every node reads from and returns partial updates to.
NODES   plan, human_review, write, critique: plain functions (state) -> update.
EDGES   fixed edges (START -> plan, write -> critique) and conditional edges
        (after review: approve or replan; after critique: revise or finish).
INTERRUPT  human_review pauses the graph with interrupt() and waits for a person.
           The checkpointer saves state so the run can resume later, even from
           another process if you use a persistent checkpointer.

    START ─► plan ─► human_review ─┬─(approved)─► write ─► critique ─┬─(good / out of revisions)─► END
               ▲                   │                ▲                │
               └────(feedback)─────┘                └───(revise)─────┘
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

MAX_REVISIONS = 2
APPROVE_TOKEN = "APPROVED"


class BriefState(TypedDict, total=False):
    topic: str
    plan: str
    feedback: str  # human feedback on the plan
    plan_approved: bool
    draft: str
    critique: str
    revisions: int
    # A reducer: instead of overwriting, each node's list is appended to this one.
    log: Annotated[list[str], operator.add]


def _ask(llm: BaseChatModel, system: str, user: str) -> str:
    reply = llm.invoke([SystemMessage(system), HumanMessage(user)])
    return reply.text.strip()


def build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None):
    def plan(state: BriefState) -> BriefState:
        user = f"Topic: {state['topic']}"
        if state.get("feedback"):
            user += f"\n\nPrevious outline:\n{state['plan']}\n\nReviewer feedback to address:\n{state['feedback']}"
        outline = _ask(
            llm,
            "You plan short technical briefs. Reply with an outline of 3 to 5 bullet points, "
            "one line each, and nothing else.",
            user,
        )
        return {"plan": outline, "log": ["plan"]}

    def human_review(state: BriefState) -> BriefState:
        # Execution stops here. The value is shown to the caller; whatever the caller
        # passes back with Command(resume=...) becomes the return value of interrupt().
        # On resume, LangGraph re-runs this node from the top, so keep the code before
        # interrupt() free of side effects.
        decision = interrupt({"question": "Approve this outline?", "plan": state["plan"]})
        if decision.get("approved"):
            return {"plan_approved": True, "feedback": "", "log": ["human_review:approved"]}
        return {
            "plan_approved": False,
            "feedback": decision.get("feedback", "Try again."),
            "log": ["human_review:rejected"],
        }

    def write(state: BriefState) -> BriefState:
        user = f"Topic: {state['topic']}\n\nOutline:\n{state['plan']}"
        if state.get("critique"):
            user += f"\n\nPrevious draft:\n{state['draft']}\n\nEditor's notes to fix:\n{state['critique']}"
        draft = _ask(
            llm,
            "You write clear, concrete technical briefs of about 250 words in Markdown, "
            "following the outline you are given.",
            user,
        )
        return {"draft": draft, "revisions": state.get("revisions", -1) + 1, "log": ["write"]}

    def critique(state: BriefState) -> BriefState:
        notes = _ask(
            llm,
            "You are a strict editor. If the brief is accurate, follows the outline and is clear, "
            f"reply with exactly {APPROVE_TOKEN}. Otherwise reply with up to three specific, "
            "actionable fixes as bullet points.",
            f"Outline:\n{state['plan']}\n\nBrief:\n{state['draft']}",
        )
        return {"critique": notes, "log": ["critique"]}

    def after_review(state: BriefState) -> Literal["write", "plan"]:
        return "write" if state["plan_approved"] else "plan"

    def after_critique(state: BriefState) -> Literal["write", "__end__"]:
        if state["critique"].strip().upper().startswith(APPROVE_TOKEN):
            return END
        if state["revisions"] >= MAX_REVISIONS:
            return END
        return "write"

    builder = StateGraph(BriefState)
    builder.add_node("plan", plan)
    builder.add_node("human_review", human_review)
    builder.add_node("write", write)
    builder.add_node("critique", critique)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "human_review")
    builder.add_conditional_edges("human_review", after_review)
    builder.add_edge("write", "critique")
    builder.add_conditional_edges("critique", after_critique)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def claude(model: str = "claude-opus-5-5", effort: str = "medium") -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # Opus 5.5 always thinks adaptively; effort is the knob for how much.
    return ChatAnthropic(model=model, max_tokens=16000, output_config={"effort": effort})
