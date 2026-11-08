# langgraph-agent

Day 2: LangGraph tutorial covering state, nodes, edges and interrupts.

A brief-writing agent. Claude drafts an outline, **you** approve it or send feedback (an interrupt), then Claude writes the brief and edits its own work in a loop until an editor pass approves it or it runs out of revisions.

```
START ─► plan ─► human_review ─┬─(approved)─► write ─► critique ─┬─(approved / max revisions)─► END
           ▲                   │                ▲                │
           └────(feedback)─────┘                └───(revise)─────┘
```

## Run it

```sh
export ANTHROPIC_API_KEY=...
uv sync
uv run brief "How prompt caching cuts LLM costs"
```

At the outline prompt, press Enter or `y` to approve, or type feedback ("add a section on cache invalidation") to send it back to the planner.

`uv run brief x --save-graph graph.mmd` writes the graph as Mermaid (paste into https://mermaid.live).

Options: `--model` (default `claude-opus-5-5`), `--effort` (default `medium`), or the `BRIEF_MODEL` / `BRIEF_EFFORT` env vars.

## Where each concept lives (`src/langgraph_agent/graph.py`)

| Concept | In this project |
|---|---|
| **State** | `BriefState`, a `TypedDict`. Nodes return only the keys they change; LangGraph merges them in. `log` uses a **reducer** (`Annotated[list, operator.add]`) so updates append instead of overwrite. |
| **Nodes** | `plan`, `human_review`, `write`, `critique`: plain functions `state -> partial update`. |
| **Edges** | Fixed: `START → plan`, `plan → human_review`, `write → critique`. Conditional: `after_review` picks `write` or `plan`; `after_critique` picks `write` or `END`. The `Literal[...]` return types tell LangGraph the possible targets (used for drawing the graph). |
| **Interrupts** | `human_review` calls `interrupt(payload)`. The run stops, the payload is visible in `graph.get_state(config).tasks[0].interrupts`, and `graph.invoke(Command(resume=value), config)` continues, making `interrupt()` return `value`. The node re-runs from the top on resume. |
| **Checkpointer** | `InMemorySaver` stores state per `thread_id` after every step; interrupts require one. Swap in `SqliteSaver` / `PostgresSaver` (`langgraph-checkpoint-sqlite` / `-postgres`) to resume a paused run from a different process. |

The CLI (`cli.py`) shows the client side: `graph.stream(..., stream_mode="updates")` to print each node as it finishes, `get_state().next` to tell "paused" from "done", and `Command(resume=...)` to continue.

## Tests

```sh
uv run pytest
```

`FakeListChatModel` replays scripted replies, so the tests cover the approve path, the reject-and-replan loop, the revision cap and thread isolation without calling the API.

## Things to try

- Add a `research` node before `plan` that gives Claude a web search tool.
- Persist with `SqliteSaver` and add a `--resume <thread-id>` flag.
- Use `interrupt` a second time to let the human approve the final draft.
- Run `graph.get_state_history(config)` to see every checkpoint, then replay from one ("time travel").
