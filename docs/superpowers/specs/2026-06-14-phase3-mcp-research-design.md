# Phase 3: MCP-backed Research — Design

**Date:** 2026-06-14
**Status:** Approved (design phase)
**Builds on:** Phase 1 (single-agent loop) and Phase 2 (dual-agent orchestration), both merged.

## 1. Summary

Give the worker access to the user's MCP tools (web search, fetch, anything in
their `mcp.json`) through a single `research` verb. The verb is backed by LM
Studio's **native agentic endpoint** `POST /api/v1/chat`, which runs MCP servers
**server-side** (tool discovery + execution + loop) and returns a synthesized
answer. We write **no MCP protocol code** — LM Studio is the MCP client.

### Why this approach (empirically chosen)

A live probe on the user's rig established:

- Native tool-calling is reliable on both models (9B and 4B each emitted valid
  `tool_calls` 3/3, grammar-constrained by llama.cpp).
- The OpenAI-compatible `/v1/chat/completions` (what our engine uses) does **not**
  run MCP — a bare "search the web" request hallucinated.
- LM Studio's **native `/api/v1/chat`** with `integrations: ["mcp/exa"]` and a
  Bearer token **did** run `web_search_exa` server-side and returned real,
  current results. The OpenAI `/v1/responses` endpoint ignored `integrations`.

So MCP execution is an LM-Studio-native feature on `/api/v1/chat`. Phase 3 adds a
thin client to that endpoint rather than reimplementing MCP.

## 2. Goals / Non-Goals

**Goals**
- A worker `research` verb that answers a natural-language query using whatever
  MCP servers the user configured in `mcp.json`.
- Zero MCP protocol implementation on our side; LM Studio does discovery +
  execution + the tool loop.
- User-extensible: adding a server to `mcp.json` makes it available with no code
  change.
- Keep the worker on the proven `::action` text protocol — no native
  tool-calling mixed into our loop.
- Secrets (the LM Studio API token) come from the environment, never committed.

**Non-Goals**
- No per-tool granular verbs (one `research` verb covers all configured tools).
- No ephemeral per-request MCP servers (only `mcp.json`-configured servers).
- No research access for the dominant (coordination-only; the worker does work).
- No change to the file/command tools, the coordination loop, or the planner.

## 3. Architecture

LM Studio exposes three API families; only the **native** one carries
`integrations`/MCP:

| Family | Endpoint(s) | MCP? |
|--------|-------------|------|
| Native | `POST /api/v1/chat` | **Yes** (`integrations`) |
| OpenAI | `/v1/chat/completions`, `/v1/responses`, … | No |
| Anthropic | `/v1/messages` | No |

Our engine keeps using the OpenAI-compatible `/v1/chat/completions` for the
agent loops; Phase 3 adds one call to the native endpoint for research.

New components (each small, single-purpose, independently testable):

- **`mcp_research.py`** — `McpResearcher`:
  - `McpResearcher(base_url, token, model, integrations, http_client=None, timeout=180.0)`.
  - `base_url` is the server root (e.g. `http://localhost:1234`); the call targets
    `{base_url}/api/v1/chat`.
  - `integrations` is a list like `["mcp/exa"]`.
  - `async research(query: str) -> str`: POST
    `{"model": model, "input": query, "integrations": integrations}` with header
    `Authorization: Bearer {token}`; `raise_for_status()`; parse the `output`
    array — return the text of the final `type == "message"` entry, prefixed with
    a one-line note of which tools ran (from the `type == "tool_call"` entries),
    e.g. `"[used: web_search_exa]\n<answer>"`.
  - Tolerant message extraction: a message's content may be a plain string or a
    list of parts (`{"type": "...text...", "text": ...}`); handle both, fall back
    to `str(output)` if the shape is unexpected.

- **`composite_registry.py`** — `CompositeRegistry`:
  - `CompositeRegistry(tool_registry, researcher)`.
  - `async execute(action) -> tuple[str, str]`: if `action.verb == "research"`,
    require a `query` arg (or use the body), `await researcher.research(query)`,
    return `("ok", answer)`; on error return `("error", str(exc))`. Otherwise
    delegate to the synchronous `tool_registry.execute(action)` (file/command).
  - The Phase 2 await-tolerant agent loop calls this directly.

- **`mcp.json` enumeration** — a helper (in `mcp_research.py` or `config.py`)
  that reads `Config.mcp_json_path`, parses the `mcpServers` object, and returns
  `["mcp/<id>" for id in servers]`. Missing file or no servers → empty list.

## 4. Control flow (a research call)

1. Worker emits `::action research` / `query: <text>` / `::end`.
2. `CompositeRegistry.execute` routes it to `McpResearcher.research(query)`.
3. `McpResearcher` POSTs to `/api/v1/chat` with the query + all `mcp.json`
   integrations + Bearer token.
4. LM Studio discovers the servers' tools, calls them, loops, and returns
   `output: [tool_call…, message]`.
5. We extract the final message text (+ tool note) and hand it back as the
   `::result`. The worker continues its normal loop.

## 5. Configuration & secrets

Extend `Config` (backward-compatible):
- `lmstudio_native_url: str = "http://localhost:1234"` (server root for the
  native endpoint).
- `research_model: str = ""` (defaults to the worker model when empty).
- `research_timeout: float = 180.0`.

The **LM Studio API token** is read from the `LMSTUDIO_TOKEN` environment
variable at runtime (in `cli.py`), never stored in `Config` or committed —
mirroring the `GEMINI_API_KEY` handling.

**Enablement:** research is wired into the worker only when a token is present
**and** `mcp.json` enumerates ≥1 server. Otherwise the worker is built with the
plain `ToolRegistry` (no `research` verb) and everything else runs unchanged.

The `WORKER_PROMPT` gains a `research` entry **only when** research is enabled,
with a concrete example:
```
::action research
query: latest NVIDIA data-center news with sources
::end
```

## 6. Error handling

| Scenario | Handling |
|----------|----------|
| `LMSTUDIO_TOKEN` unset or `mcp.json` empty | `research` verb not registered; worker runs with file/command only. |
| Native call 4xx/5xx (bad token, "allow mcp.json" toggle off, etc.) | `McpResearcher` surfaces it; `CompositeRegistry` returns `("error", message)` to the worker. Run continues. |
| Unexpected `output` shape | Tolerant extractor falls back to a stringified output rather than crashing. |
| Unknown verb / missing `query` | `("error", message)`, same as other tools. |

## 7. Testing

- **`mcp_research.py`** via `httpx.MockTransport`: assert the request targets
  `/api/v1/chat`, includes `input`, `integrations`, and the `Authorization:
  Bearer` header; parse a mocked `output` (tool_call + message) into the final
  answer with the tool note; cover the string-content and list-content message
  shapes; cover an HTTP error path. No live calls; no real token.
- **`CompositeRegistry`**: `research` routes to a fake researcher and returns its
  answer; file/command verbs route to a real `ToolRegistry` (tmp_path); unknown
  verb → error; researcher exception → `("error", …)`.
- **mcp.json enumeration**: a sample config with one and with multiple servers →
  correct `["mcp/<id>", …]`; missing file → `[]`.
- **Optional live smoke** (excluded from the unit suite): real LM Studio +
  `LMSTUDIO_TOKEN` + Exa on a trivial query.

## 8. Stack & Phasing

**Stack:** unchanged — Python 3.11+, `httpx`, `pytest` + `pytest-asyncio`. The
native call is plain `httpx` (no LM Studio SDK dependency).

**Build order (single plan):**
1. `mcp_research.py` (researcher + mcp.json enumeration).
2. `composite_registry.py`.
3. `config.py` additions.
4. Wire into `cli.py` worker_factory + conditional `WORKER_PROMPT` research line.

## 9. Risks

- **Native API shape drift.** The `/api/v1/chat` response shape is newer and may
  vary by LM Studio version; the tolerant extractor and an explicit
  `raise_for_status` keep failures graceful and debuggable.
- **Setting/version dependency.** Requires LM Studio 0.4.0+ with "allow calling
  servers from mcp.json" enabled and a valid token. When unmet, research is
  simply disabled — the engine still runs.
- **Coarse granularity.** One `research` verb means the model can't target a
  specific tool. Acceptable for this phase; per-tool verbs can come later if
  needed.
