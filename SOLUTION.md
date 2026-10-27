# TAM Ticket Triage Agent

## Design

The agent uses a native tool-calling ReAct loop: the model chooses the next lookup, receives the tool observation, and calls `finish` only when it has enough evidence. It is not a fixed lookup sequence. The loop is capped at eight model turns; the cap requests one final best-effort `finish` while retaining the account and KB state already found.

```
ticket -> LLM chooses tools -> code executes/caches tools -> observations -> LLM finish
              |                    |
              +-- resolve, health, history, KB/policy --+-> validated JSON
```

The shared `Context` builds indexes once: domain/name account lookup, per-account ticket history and usage trends, a KB BM25 index, and an alias map derived from the ticket corpus and KB (including `CLM -> Contracts`, `eProc -> eProcurement`, `analytics -> Spend Analytics`, and `supplier portal -> Supplier Management`). The tools are `resolve_account`, `get_account_health`, `get_ticket_history`, `search_kb`, `get_kb_article`, `get_policy`, and `finish`.

The agent has code-owned trust boundaries. Only `resolve_account` can establish `account_id`; the code pins health/history queries to that resolved ID and supplies account-resolution inputs from the ticket. Every native tool call in a model message is executed (including parallel calls), receives a matching `tool_call_id` response, and identical normalized calls use the cached result. The code owns the trace and appends its `finish` step.

KB snippets are never citations. `kb_references` is limited to full KB/policy articles actually read. Customer-facing operational instructions require an exact source sentence from a read article; that audit evidence is written to `verification/<ticket>.json`, while the fixed submission JSON remains schema-compatible. Output guards remove links and internal account thresholds, cap confidence at 0.85, and annotate detected prompt-injection attempts in the internal note. Raw invalid finishes are returned to the model for correction instead of being silently normalized.

Policy is grounded in KB-2914, KB-5930, and KB-6401. Code validates duplicate candidates against the current ticket's preceding seven-day window, preserves the original ticket priority, and prevents re-escalation. It calculates the account-risk override only from resolved account/usage tool results. The Groq client has model failover from the configured Qwen model to `openai/gpt-oss-120b` on a daily-token-quota response.

## Evaluation before production

- Require schema-valid outputs and inspect grounded evidence, not just `check_outputs.py`.
- Compare category/priority/routing/escalation against TAM-labelled tickets; treat missed P1/security cases as highest severity.
- Measure duplicate precision/recall using labelled pairs and policy-window tests.
- Audit KB citation precision and every customer-reply instruction's source sentence.
- Test account isolation, prompt-injection resistance, and tool-call protocol handling.
- Monitor completion rate, model/tool calls, latency, rate-limit failover, and cost per ticket.

## Scale and limitations

At 10x volume, indexes remain practical in memory but lexical BM25 retrieval should become hybrid retrieval (embeddings + BM25/RRF + reranking). Move ticket/account lookups to DuckDB or a service, precompute embeddings, process tickets asynchronously under a provider rate limiter, and persist checkpoints. The current limitations are lexical retrieval, model-dependent decision quality, sequential `run.py` execution, and only model-level (not cross-provider) automatic failover.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\validate_data.py
.\.venv\Scripts\python.exe run.py --set review
.\.venv\Scripts\python.exe scripts\check_outputs.py
```

Configure the provider and key only in `.env` (ignored by Git). Review `verification/` beside the 10 generated JSON outputs before submission.
