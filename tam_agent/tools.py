"""Tool implementations for the TAM triage agent.

Each function is called by the agent loop when the LLM chooses that tool.
All tools operate on pre-built indexes from retrieval.py for speed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import Context

AS_OF_DATE = "2026-10-20"


def _as_of() -> datetime:
    return datetime.strptime(AS_OF_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _days_until(date_str: str) -> int | None:
    """Return days from AS_OF_DATE to date_str (positive = future). None if unparseable."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
            return (dt - _as_of()).days
        except ValueError:
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: resolve_account
# ══════════════════════════════════════════════════════════════════════════════

def resolve_account(ctx: "Context", requester_domain: str = "", company_name_raw: str = "") -> dict:
    """Resolve a ticket to a known account.

    Always try domain first (always resolves per data spec).
    Falls back to fuzzy company name match if domain is missing.
    Returns account record + computed signals.
    """
    account = None
    method = ""

    if requester_domain:
        account = ctx.account_index.by_domain(requester_domain)
        method = f"domain:{requester_domain}"

    if account is None and company_name_raw:
        account = ctx.account_index.fuzzy_by_name(company_name_raw)
        method = f"fuzzy_name:{company_name_raw}"

    if account is None:
        return {"found": False, "method": method, "error": "Could not resolve account"}

    acc_id = account.get("account_id", "")
    tier = account.get("tier", "Standard")
    renewal_str = account.get("renewal_date", "")
    escalations = int(account.get("open_escalations", "0") or "0")
    nps_raw = account.get("nps_last", "")
    arr_band = account.get("arr_band", "")
    modules = account.get("modules_licensed", "")
    last_qbr = account.get("last_qbr_date", "")

    # Compute renewal signal
    days_to_renewal = _days_until(renewal_str) if renewal_str else None

    signals = []
    if days_to_renewal is not None:
        if days_to_renewal <= 0:
            signals.append(f"Renewal OVERDUE ({renewal_str})")
        elif days_to_renewal <= 14:
            signals.append(f"Renewal critically soon: {days_to_renewal} days ({renewal_str})")
        elif days_to_renewal <= 30:
            signals.append(f"Renewal in {days_to_renewal} days ({renewal_str}) — imminent")
        elif days_to_renewal <= 90:
            signals.append(f"Renewal in {days_to_renewal} days ({renewal_str}) — within risk window")

    if escalations >= 2:
        signals.append(f"{escalations} open escalations — high risk")
    elif escalations == 1:
        signals.append("1 open escalation")

    try:
        nps = float(nps_raw)
        if nps <= 4:
            signals.append(f"NPS {nps:.0f} — at-risk customer")
        elif nps <= 6:
            signals.append(f"NPS {nps:.0f} — passive")
    except (ValueError, TypeError):
        pass

    if tier in ("Enterprise", "Premium"):
        signals.append(f"{tier} tier account")

    return {
        "found": True,
        "method": method,
        "account_id": acc_id,
        "company_name": account.get("company_name", ""),
        "domain": account.get("domain", ""),
        "tier": tier,
        "arr_band": arr_band,
        "renewal_date": renewal_str,
        "days_to_renewal": days_to_renewal,
        "open_escalations": escalations,
        "nps_last": nps_raw,
        "modules_licensed": modules,
        "last_qbr_date": last_qbr,
        "region": account.get("region", ""),
        "industry": account.get("industry", ""),
        "account_signals": signals,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: get_account_health
# ══════════════════════════════════════════════════════════════════════════════

def get_account_health(ctx: "Context", account_id: str, months: int = 3) -> dict:
    """Return usage trend for the last N months plus risk signals."""
    trend = ctx.usage_index.trend_summary(account_id, n=months)
    if not trend.get("available"):
        return {"account_id": account_id, "available": False, "summary": "No usage data."}

    lines = [f"Usage trend ({', '.join(str(m) for m in trend.get('months', []))})"]
    signals = []

    for metric_key, label in [
        ("active_users", "Active users"),
        ("login_rate_pct", "Login rate %"),
        ("documents_processed", "Docs processed"),
        ("tickets_opened", "Tickets opened"),
    ]:
        m = trend.get(metric_key)
        if isinstance(m, dict):
            direction_emoji = {"declining": "↓", "improving": "↑", "stable": "→"}.get(m["direction"], "")
            lines.append(
                f"  {label}: {m.get('latest')} {direction_emoji} ({m.get('pct_change', 0):+.1f}% over {months}mo)"
            )
            if m["direction"] == "declining" and abs(m.get("pct_change", 0)) >= 15:
                signals.append(f"{label} declining {abs(m.get('pct_change', 0)):.0f}% over {months} months")

    risk_override = trend.get("risk_override_usage", False)
    if risk_override:
        signals.append("Usage decline >25% — account risk override may apply (KB-2914)")

    return {
        "account_id": account_id,
        "available": True,
        "months": trend.get("months"),
        "active_users": trend.get("active_users"),
        "login_rate_pct": trend.get("login_rate_pct"),
        "documents_processed": trend.get("documents_processed"),
        "tickets_opened": trend.get("tickets_opened"),
        "risk_override_usage": risk_override,
        "health_signals": signals,
        "summary": "\n".join(lines),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3: get_ticket_history
# ══════════════════════════════════════════════════════════════════════════════

def get_ticket_history(
    ctx: "Context",
    account_id: str,
    current_ticket_id: str = "",
    module: str = "",
    error_code: str = "",
    subject: str = "",
    current_created_at: str = "",
    days_back: int = 7,
) -> dict:
    """Return recent ticket history for an account and check for duplicates.

    Per KB-5930: duplicate = same account, same module + same error/symptom, within 7 days, open/pending.
    """
    recent = ctx.ticket_index.by_account(account_id, limit=30)

    # Recent summary
    summary_lines = []
    from datetime import timedelta
    as_of = _as_of()
    last_30d = [
        r for r in recent
        if _parse_dt(r.get("created_at", "")) and (as_of - _parse_dt(r.get("created_at", ""))) <= timedelta(days=30)
    ]

    summary_lines.append(f"Total tickets on file: {len(recent)}")
    summary_lines.append(f"Last 30 days: {len(last_30d)} tickets")

    # Module breakdown
    modules_seen: dict[str, int] = {}
    for r in last_30d:
        m = r.get("module", "unknown")
        modules_seen[m] = modules_seen.get(m, 0) + 1
    if modules_seen:
        top = sorted(modules_seen.items(), key=lambda x: -x[1])[:5]
        summary_lines.append("Top modules: " + ", ".join(f"{m}({c})" for m, c in top))

    # Last 5 tickets preview
    recent_preview = []
    for r in sorted(recent, key=lambda x: x.get("created_at", ""), reverse=True)[:5]:
        recent_preview.append({
            "ticket_id": r.get("ticket_id"),
            "created_at": r.get("created_at"),
            "module": r.get("module"),
            "subject": r.get("subject", "")[:80],
            "status": r.get("status"),
            "error_code": r.get("error_code"),
            "priority": r.get("customer_priority") or "P3",
        })

    # Duplicate check
    duplicates = []
    if current_ticket_id or module or error_code or subject:
        duplicates = ctx.ticket_index.find_duplicates(
            account_id=account_id,
            module=module,
            error_code=error_code,
            subject=subject,
            current_ticket_id=current_ticket_id,
            current_created_at=current_created_at,
            days_back=days_back,
        )

    dup_info = []
    for d in duplicates[:3]:
        dup_info.append({
            "ticket_id": d.get("ticket_id"),
            "created_at": d.get("created_at"),
            "module": d.get("module"),
            "subject": d.get("subject", "")[:80],
            "status": d.get("status"),
            "error_code": d.get("error_code"),
            "priority": d.get("customer_priority") or "P3",
        })

    return {
        "account_id": account_id,
        "total_tickets": len(recent),
        "summary": "\n".join(summary_lines),
        "recent_tickets": recent_preview,
        "potential_duplicates": dup_info,
        "duplicate_found": len(dup_info) > 0,
    }


def _parse_dt(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Tool 4: search_kb
# ══════════════════════════════════════════════════════════════════════════════

def search_kb(
    ctx: "Context",
    query: str,
    module: str = "",
    type_filter: str = "",
    top_k: int = 5,
) -> dict:
    """BM25 search over KB articles. Returns top_k results with snippets.

    Excludes deprecated articles (they are followed automatically by get_kb_article).
    Use get_kb_article to fetch the full content.
    """
    results = ctx.kb_index.search(
        query=query,
        top_k=top_k,
        module_filter=module or None,
        type_filter=type_filter or None,
        exclude_deprecated=True,
    )
    return {
        "query": query,
        "results": results,
        "count": len(results),
        "note": "Call get_kb_article with an article ID to read the full content.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 5: get_kb_article
# ══════════════════════════════════════════════════════════════════════════════

def get_kb_article(ctx: "Context", article_id: str) -> dict:
    """Fetch a full KB article by ID.

    If the article is deprecated, automatically follows the superseded_by chain.
    Returns metadata + full body content.
    """
    art = ctx.kb_index.get_article(article_id, follow_superseded=True)
    if not art:
        return {"found": False, "article_id": article_id, "error": "Article not found"}

    meta = art.get("meta", {})
    result = {
        "found": True,
        "id": art.get("id"),
        "title": meta.get("title", ""),
        "module": meta.get("module", ""),
        "type": meta.get("type", ""),
        "status": meta.get("status", ""),
        "error_codes": meta.get("error_codes", []),
        "last_updated": meta.get("last_updated", ""),
        "related": meta.get("related", []),
        "body": art.get("body", ""),
    }
    if art.get("_followed_from"):
        result["note"] = f"This article supersedes {art['_followed_from']} (deprecated)."
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Tool 6: get_policy
# ══════════════════════════════════════════════════════════════════════════════

def get_policy(ctx: "Context", policy_type: str = "all") -> dict:
    """Return support policy articles (priority, routing, escalation, duplicate, security).

    policy_type: 'all' | 'priority' | 'escalation' | 'routing' | 'duplicate' | 'security'
    These are pre-cached and always current — call this early in triage.
    """
    policy_articles = ctx.kb_index.get_policy_articles()

    if not policy_articles:
        return {"found": False, "error": "No policy articles found"}

    # Filter by keyword if not 'all'
    keyword_map = {
        "priority": ["priority", "sla", "first response"],
        "escalation": ["escalat"],
        "routing": ["routing", "route", "L1", "L2", "L3"],
        "duplicate": ["duplicate", "duplicat"],
        "security": ["security", "data exposure", "data breach"],
    }

    selected = policy_articles
    if policy_type != "all":
        keywords = keyword_map.get(policy_type.lower(), [])
        if keywords:
            selected = [
                a for a in policy_articles
                if any(kw.lower() in (a.get("meta", {}).get("title", "") + a.get("body", "")).lower()
                       for kw in keywords)
            ]
        if not selected:
            selected = policy_articles  # fallback

    return {
        "policy_type": policy_type,
        "articles": [
            {
                "id": a.get("id"),
                "title": a.get("meta", {}).get("title", ""),
                "body": a.get("body", "").strip(),
            }
            for a in selected
        ],
        "count": len(selected),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 7: finish
# ══════════════════════════════════════════════════════════════════════════════

def finish(ctx: "Context", decision: dict) -> dict:
    """Validate and return the final triage decision.

    If the decision fails schema validation, returns errors for the agent to fix.
    """
    from .schema import validate
    errs = validate(decision)
    if errs:
        return {
            "valid": False,
            "errors": errs,
            "message": "Fix these schema errors and call finish again.",
        }
    return {"valid": True, "decision": decision}


# ══════════════════════════════════════════════════════════════════════════════
# Tool registry for the agent loop
# ══════════════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS = {
    "resolve_account": resolve_account,
    "get_account_health": get_account_health,
    "get_ticket_history": get_ticket_history,
    "search_kb": search_kb,
    "get_kb_article": get_kb_article,
    "get_policy": get_policy,
    "finish": finish,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_account",
            "description": "Resolve the ticket to a known account. Code supplies requester_domain and company_name_raw; you may call this with empty args. Returns tier, renewal, escalations, NPS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requester_domain": {"type": "string", "description": "Ignored; code fills from the ticket."},
                    "company_name_raw": {"type": "string", "description": "Ignored; code fills from the ticket."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_health",
            "description": "Last 3 months of usage for an account: active users, login rate, documents, tickets. Includes trend and risk_override_usage (>25% active-user decline).",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "months": {"type": "integer", "default": 3},
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_history",
            "description": "Recent tickets for an account and duplicate check. Per KB-5930 a duplicate is same account, same module, AND same error code or symptom, within 7 days, status open or pending_customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "current_ticket_id": {"type": "string"},
                    "module": {"type": "string"},
                    "error_code": {"type": "string"},
                    "subject": {"type": "string"},
                    "current_created_at": {"type": "string", "description": "Ignored; code supplies the current ticket timestamp."},
                    "days_back": {"type": "integer", "default": 7},
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "BM25 search over KB articles. Returns snippets only. You MUST call get_kb_article on any ID you will cite or use for reply steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "module": {"type": "string"},
                    "type_filter": {
                        "type": "string",
                        "description": "troubleshooting | how_to | policy | known_issue | limit | faq | feature_request",
                    },
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_kb_article",
            "description": "Read the full KB article. Follows superseded_by when deprecated. Reply steps may only come from articles you read with this tool or get_policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string"},
                },
                "required": ["article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy",
            "description": "Read support policy (KB-2914 priority/routing/escalation, KB-5930 duplicates, KB-6401 security). Call early.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_type": {
                        "type": "string",
                        "description": "all | priority | escalation | routing | duplicate | security",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final triage decision after you have resolved the account, read policy, and read the relevant KB article(s).",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "object",
                        "properties": {
                            "ticket_id": {"type": "string"},
                            "account_id": {"type": ["string", "null"]},
                            "category": {
                                "type": "string",
                                "enum": ["incident", "bug", "how_to", "feature_request", "security", "duplicate", "needs_info"],
                            },
                            "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                            "route_to": {"type": "string", "enum": ["L1_support", "L2_engineering", "security", "product"]},
                            "escalate": {"type": "boolean"},
                            "notify_tam": {"type": "boolean"},
                            "duplicate_of": {"type": ["string", "null"]},
                            "needs_clarification": {"type": "boolean"},
                            "kb_references": {"type": "array", "items": {"type": "string"}},
                            "account_signals": {"type": "array", "items": {"type": "string"}},
                            "customer_reply": {"type": "string"},
                            "reply_step_sources": {
                                "type": "array",
                                "description": "For each numbered or bulleted customer-reply step: its KB ID and an exact supporting sentence from that article. Omit only when the reply has no steps.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "kb_id": {"type": "string"},
                                        "source_sentence": {"type": "string"},
                                    },
                                    "required": ["kb_id", "source_sentence"],
                                },
                            },
                            "internal_note": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "ticket_id", "category", "priority", "route_to", "escalate",
                            "notify_tam", "needs_clarification", "customer_reply", "internal_note",
                        ],
                    },
                },
                "required": ["decision"],
            },
        },
    },
]
