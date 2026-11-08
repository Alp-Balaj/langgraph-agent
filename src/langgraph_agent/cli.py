"""Run the brief writer in the terminal.

    uv run brief "How prompt caching cuts LLM costs"
"""

import argparse
import os
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from .graph import build_graph, claude


def main() -> None:
    load_dotenv()  # pick up ANTHROPIC_API_KEY etc. from .env; real env vars win
    parser = argparse.ArgumentParser(description="Plan, approve, write and self-edit a short brief.")
    parser.add_argument("topic", help="what the brief is about")
    parser.add_argument("--model", default=os.environ.get("BRIEF_MODEL", "claude-opus-5-5"))
    parser.add_argument("--effort", default=os.environ.get("BRIEF_EFFORT", "medium"))
    parser.add_argument("--save-graph", metavar="FILE", help="write the graph as Mermaid and exit")
    args = parser.parse_args()

    graph = build_graph(claude(args.model, args.effort))
    if args.save_graph:
        with open(args.save_graph, "w") as f:
            f.write(graph.get_graph().draw_mermaid())
        print(f"Wrote {args.save_graph}")
        return

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    next_input: dict | Command = {"topic": args.topic}

    while True:
        # stream_mode="updates" yields {node_name: update} after each node finishes.
        for chunk in graph.stream(next_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    continue
                print(f"\n── {node} " + "─" * (60 - len(node)))
                if node == "plan":
                    print(update["plan"])
                elif node == "write":
                    print(f"(draft {update['revisions'] + 1} written, {len(update['draft'].split())} words)")
                elif node == "critique":
                    print(update["critique"])

        state = graph.get_state(config)
        if not state.next:  # nothing left to run: finished
            break

        # Paused at an interrupt: show its payload and collect a decision.
        payload = state.tasks[0].interrupts[0].value
        print(f"\n{payload['question']} [y / or type feedback] ", end="")
        reply = input().strip()
        approved = reply.lower() in {"", "y", "yes"}
        next_input = Command(resume={"approved": approved, "feedback": "" if approved else reply})

    final = graph.get_state(config).values
    print("\n" + "═" * 64 + "\n")
    print(final["draft"])
    print(f"\n(revisions: {final['revisions']}, path: {' → '.join(final['log'])})")
