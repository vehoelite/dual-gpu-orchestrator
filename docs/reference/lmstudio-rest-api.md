# LM Studio REST API (reference)

> Saved reference from LM Studio's developer docs. Relevant to Phase 3: MCP via
> API. See `docs/superpowers/specs/2026-06-14-phase3-mcp-research-design.md`.

LM Studio offers a powerful REST API with first-class support for local inference and model management. In addition to our native API, we provide OpenAI-compatible endpoints ([learn more](/docs/developer/openai-compat)) and Anthropic-compatible endpoints ([learn more](/docs/developer/anthropic-compat)).

## What's new

Previously, there was a [v0 REST API](/docs/developer/rest/endpoints). With LM Studio 0.4.0, we have officially released our native v1 REST API at `/api/v1/*` endpoints and recommend using it.

The v1 REST API includes enhanced features such as:

- [MCP via API](/docs/developer/core/mcp)
- [Stateful chats](/docs/developer/rest/stateful-chats)
- [Authentication](/docs/developer/core/authentication) configuration with API tokens
- Model [download](/docs/developer/rest/download), [load](/docs/developer/rest/load) and [unload](/docs/developer/rest/unload) endpoints

## Supported endpoints

The following endpoints are available in LM Studio's v1 REST API.

| Endpoint | Method | Docs |
|----------|--------|------|
| `/api/v1/chat` | POST | Chat |
| `/api/v1/models` | GET | List Models |
| `/api/v1/models/load` | POST | Load |
| `/api/v1/models/unload` | POST | Unload |
| `/api/v1/models/download` | POST | Download |
| `/api/v1/models/download/status` | GET | Download Status |

## Inference endpoint comparison

The table below compares the features of LM Studio's `/api/v1/chat` endpoint with OpenAI-compatible and Anthropic-compatible inference endpoints.

| Feature | `/api/v1/chat` | `/v1/responses` | `/v1/chat/completions` | `/v1/messages` |
|---------|:--------------:|:---------------:|:----------------------:|:--------------:|
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Stateful chat | ✅ | ✅ | ❌ | ❌ |
| Remote MCPs | ✅ | ✅ | ❌ | ❌ |
| MCPs you have in LM Studio | ✅ | ✅ | ❌ | ❌ |
| Custom tools | ❌ | ✅ | ✅ | ✅ |
| Include assistant messages in the request | ❌ | ✅ | ✅ | ✅ |
| Model load streaming events | ✅ | ❌ | ❌ | ❌ |
| Prompt processing streaming events | ✅ | ❌ | ❌ | ❌ |
| Specify context length in the request | ✅ | ❌ | ❌ | ❌ |

---

Please report bugs by opening an issue on [Github](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues).

---

## Note for our project (not from the LM Studio docs)

The comparison table shows MCP works on **both** `/api/v1/chat` (native) **and**
`/v1/responses` (OpenAI-compat). Our live probe found `/v1/responses` ignored the
request and hallucinated — because MCP on `/v1/responses` uses the **OpenAI MCP
tool format** (an entry in the `tools` array), not LM Studio's native
`integrations` field. We sent `integrations`, which `/v1/responses` doesn't
recognize. Our design uses the **native `/api/v1/chat` + `integrations`** path,
which the probe confirmed runs Exa server-side and returns live results.

Two endpoint families also do **not** support custom (client-defined) tools on
`/api/v1/chat` (see "Custom tools" ❌). That's fine for us: our file/command and
coordination tools run through our own `::action` text protocol on
`/v1/chat/completions`, and `/api/v1/chat` is used *only* for MCP-backed research.
