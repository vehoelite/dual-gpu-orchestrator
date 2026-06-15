# MCP via the LM Studio API (reference)

> Requires **LM Studio 0.4.0+**. Saved reference from LM Studio's developer docs.
> This is the mechanism Phase 3 uses (native `/api/v1/chat` + `integrations`).
> See `docs/superpowers/specs/2026-06-14-phase3-mcp-research-design.md`.

LM Studio supports Model Context Protocol (MCP) usage via API. MCP allows models to interact with external tools and services through standardized servers.

## How it works

MCP servers provide tools that models can call during chat requests. You can enable MCP servers in two ways: **ephemeral** servers defined per-request, or **pre-configured** servers in your `mcp.json` file.

## Ephemeral vs mcp.json servers

| Feature | Ephemeral | mcp.json |
|---------|-----------|----------|
| How to specify in request | `integrations` → `"type": "ephemeral_mcp"` | `integrations` → `"type": "plugin"` (or the string `"mcp/<id>"`) |
| Configuration | Only defined per-request | Pre-configured in `mcp.json` |
| Use case | One-off requests, remote MCP tool execution | Servers that require `command`, frequently used servers |
| Server ID | `server_label` in the integration | `id` (e.g. `mcp/playwright`) in the integration |
| Custom headers | Supported via `headers` field | Configured in `mcp.json` |
| Required setting | "Allow per-request MCPs" | "Allow calling servers from mcp.json" |

Both settings live in LM Studio → Server Settings. Auth is a Bearer token
(`Authorization: Bearer $LM_API_TOKEN`).

## mcp.json servers (the path we use)

Reference a pre-configured server by id. Requires "Allow calling servers from mcp.json".

```python
import os, requests, json

response = requests.post(
    "http://localhost:1234/api/v1/chat",
    headers={
        "Authorization": f"Bearer {os.environ['LM_API_TOKEN']}",
        "Content-Type": "application/json",
    },
    json={
        "model": "ibm/granite-4-micro",
        "input": "Open lmstudio.ai",
        "integrations": ["mcp/playwright"],
        "context_length": 8000,
        "temperature": 0,
    },
)
print(json.dumps(response.json(), indent=2))
```

Response — `output` is a list of `reasoning` / `message` / `tool_call` items;
the **final `type == "message"` entry** holds the answer:

```json
{
  "model_instance_id": "ibm/granite-4-micro",
  "output": [
    { "type": "reasoning", "content": "..." },
    { "type": "message", "content": "..." },
    {
      "type": "tool_call",
      "tool": "browser_navigate",
      "arguments": { "url": "https://..." },
      "output": "...",
      "provider_info": { "plugin_id": "mcp/playwright", "type": "plugin" }
    },
    { "type": "reasoning", "content": "..." },
    { "type": "message", "content": "The YouTube video page for ..." }
  ],
  "stats": { "input_tokens": 2614, "total_output_tokens": 594 },
  "response_id": "resp_..."
}
```

## Ephemeral servers (per-request)

Defined on the fly; good for testing or remote servers you don't want to pre-configure. Requires "Allow per-request MCPs".

```python
import os, requests, json

response = requests.post(
    "http://localhost:1234/api/v1/chat",
    headers={
        "Authorization": f"Bearer {os.environ['LM_API_TOKEN']}",
        "Content-Type": "application/json",
    },
    json={
        "model": "ibm/granite-4-micro",
        "input": "What is the top trending model on hugging face?",
        "integrations": [
            {
                "type": "ephemeral_mcp",
                "server_label": "huggingface",
                "server_url": "https://huggingface.co/mcp",
                "allowed_tools": ["model_search"],
            }
        ],
        "context_length": 8000,
    },
)
print(json.dumps(response.json(), indent=2))
```

Response (abridged) — note `provider_info.type == "ephemeral_mcp"`:

```json
{
  "model_instance_id": "ibm/granite-4-micro",
  "output": [
    { "type": "reasoning", "content": "..." },
    { "type": "message", "content": "..." },
    {
      "type": "tool_call",
      "tool": "model_search",
      "arguments": { "sort": "trendingScore", "limit": 1 },
      "output": "...",
      "provider_info": { "server_label": "huggingface", "type": "ephemeral_mcp" }
    },
    { "type": "reasoning", "content": "\n" },
    { "type": "message", "content": "The top trending model is ..." }
  ],
  "response_id": "resp_..."
}
```

## Restricting tool access

For either server type, `allowed_tools` limits which tools the model may call.
Fewer tool definitions also speeds prompt processing. If omitted, **all** of the
server's tools are available.

## Custom headers (ephemeral servers)

Ephemeral servers needing their own auth can pass `headers`:

```json
"integrations": [
  {
    "type": "ephemeral_mcp",
    "server_label": "huggingface",
    "server_url": "https://huggingface.co/mcp",
    "allowed_tools": ["model_search"],
    "headers": { "Authorization": "Bearer <YOUR_HF_TOKEN>" }
  }
]
```

---

## Relevance to our design

- We use the **mcp.json** path: `integrations: ["mcp/<id>"]` for each server in
  the user's `mcp.json` (the worker's `research` verb). Requires the "Allow
  calling servers from mcp.json" setting + Bearer token (`LMSTUDIO_TOKEN`).
- `McpResearcher` parses `output` and returns the **last `type == "message"`**
  entry's `content` (interim `reasoning`/`message`/`tool_call` items precede it),
  optionally noting which `tool_call`s ran.
- `context_length` and `allowed_tools` are available optional knobs we can add
  later; the initial `research` verb keeps it simple (no `allowed_tools` → all
  configured tools).
- Ephemeral servers are out of scope for Phase 3 (mcp.json only), but the same
  endpoint supports them if we want per-request remote servers later.
