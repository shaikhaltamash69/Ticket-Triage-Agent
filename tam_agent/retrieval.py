"""Retrieval and indexing layer.

Built once at Context init time, reused for every ticket.
Provides:
  - Domain → account lookup (O(1))
  - Fuzzy company name → account_id (for blank account_id tickets)
  - Account → ticket history index
  - Account → usage metrics index
  - BM25 KB search
  - Policy article cache
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

# ── Today's reference date (from data README) ────────────────────────────────
AS_OF_DATE = "2026-10-20"

# Seed mappings required by the task (ticket slang -> canonical KB module).
_SEED_ALIASES = {
    "clm": "contracts",
    "contract": "contracts",
    "contracts": "contracts",
    "eproc": "eprocurement",
    "eprocurement": "eprocurement",
    "e procurement": "eprocurement",
    "analytics": "spend analytics",
    "spend analytics": "spend analytics",
    "spend": "spend analytics",
    "supplier portal": "supplier management",
    "supplier management": "supplier management",
    "einvoice": "einvoice",
    "e invoice": "einvoice",
    "einv": "einvoice",
    "sourcing": "sourcing",
    "rfx": "sourcing",
    "esourcing": "sourcing",
    "integrations": "integrations",
    "integration": "integrations",
    "platform": "platform",
}


def build_module_alias_map(tickets_df: pd.DataFrame | None, articles: list[dict]) -> dict[str, str]:
    """Map normalised module aliases from tickets.csv + KB modules to canonical names."""
    canon: set[str] = set()
    for art in articles or []:
        n = _normalise(str((art.get("meta") or {}).get("module") or ""))
        if n and n != "all":
            canon.add(n)

    mapping: dict[str, str] = {c: c for c in canon}
    for alias, target in _SEED_ALIASES.items():
        mapping[alias] = target if target in canon else target

    if tickets_df is not None and "module" in tickets_df.columns:
        for raw in tickets_df["module"].dropna().unique():
            n = _normalise(str(raw))
            if not n or n in mapping:
                continue
            mapped = None
            for alias, target in _SEED_ALIASES.items():
                if n == alias or alias in n or n in alias:
                    mapped = mapping.get(alias, target)
                    break
            if not mapped:
                for c in canon:
                    if n in c or c in n:
                        mapped = c
                        break
            mapping[n] = mapped or n
    return mapping


def canonical_module(raw: str, alias_map: dict[str, str] | None) -> str:
    n = _normalise(raw or "")
    if not n:
        return ""
    if alias_map and n in alias_map:
        return alias_map[n]
    if n in _SEED_ALIASES:
        return _SEED_ALIASES[n]
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Text normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenise(text: str) -> list[str]:
    return _normalise(text).split()


def _fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


# ══════════════════════════════════════════════════════════════════════════════
# Account index
# ══════════════════════════════════════════════════════════════════════════════

class AccountIndex:
    """Fast account lookups keyed by domain or normalised company name."""

    def __init__(self, accounts_df: pd.DataFrame):
        self._df = accounts_df
        # domain → row dict
        self._domain_map: dict[str, dict] = {}
        # account_id → row dict
        self._id_map: dict[str, dict] = {}
        # normalised company name → account_id  (for fuzzy fallback)
        self._name_to_id: dict[str, str] = {}

        for _, row in accounts_df.iterrows():
            rec = row.to_dict()
            domain = rec.get("domain", "").strip().lower()
            acc_id = rec.get("account_id", "").strip()
            name = rec.get("company_name", "")
            if domain:
                self._domain_map[domain] = rec
            if acc_id:
                self._id_map[acc_id] = rec
            if name:
                self._name_to_id[_normalise(name)] = acc_id

    def by_domain(self, domain: str) -> dict | None:
        return self._domain_map.get(domain.strip().lower())

    def by_id(self, account_id: str) -> dict | None:
        return self._id_map.get(account_id.strip())

    def fuzzy_by_name(self, raw_name: str, threshold: float = 0.70) -> dict | None:
        """Return best-matching account by company name (fuzzy)."""
        best_score, best_id = 0.0, None
        norm = _normalise(raw_name)
        for stored_name, acc_id in self._name_to_id.items():
            score = _fuzzy_score(norm, stored_name)
            if score > best_score:
                best_score, best_id = score, acc_id
        if best_score >= threshold and best_id:
            return self._id_map.get(best_id)
        return None

    def all_names(self) -> list[str]:
        return list(self._name_to_id.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Ticket history index
# ══════════════════════════════════════════════════════════════════════════════

class TicketHistoryIndex:
    """Fast ticket lookups per account, with duplicate detection helpers."""

    def __init__(self, tickets_df: pd.DataFrame, alias_map: dict[str, str] | None = None):
        self._df = tickets_df
        self._alias_map = alias_map or {}
        # account_id → list[dict]
        self._by_account: dict[str, list[dict]] = {}
        # ticket_id → dict
        self._by_id: dict[str, dict] = {}

        for _, row in tickets_df.iterrows():
            rec = row.to_dict()
            acc_id = rec.get("account_id", "").strip()
            tid = rec.get("ticket_id", "").strip()
            if acc_id:
                self._by_account.setdefault(acc_id, []).append(rec)
            if tid:
                self._by_id[tid] = rec

    def by_account(self, account_id: str, limit: int = 50) -> list[dict]:
        return self._by_account.get(account_id.strip(), [])[-limit:]

    def by_id(self, ticket_id: str) -> dict | None:
        return self._by_id.get(ticket_id.strip())

    def find_duplicates(
        self,
        account_id: str,
        module: str,
        error_code: str,
        subject: str,
        current_ticket_id: str,
        current_created_at: str = "",
        days_back: int = 7,
    ) -> list[dict]:
        """Find potential duplicate tickets per KB-5930 policy.

        Same account, same (canonical) module, AND same error code or symptom,
        created within the previous 7 days, status open or pending_customer.
        """
        from datetime import datetime, timedelta, timezone
        # The policy window is relative to the current ticket. A ticket opened
        # after it can never be the "original" duplicate.
        try:
            current_time = datetime.fromisoformat(current_created_at).replace(tzinfo=timezone.utc)
        except Exception:
            current_time = datetime.strptime(AS_OF_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cutoff = current_time - timedelta(days=days_back)

        target_mod = canonical_module(module, self._alias_map)
        error_code = (error_code or "").strip()
        candidates = []
        for rec in self._by_account.get(account_id.strip(), []):
            if rec.get("ticket_id") == current_ticket_id:
                continue
            if rec.get("status", "") not in ("open", "pending_customer"):
                continue
            try:
                created = datetime.fromisoformat(rec.get("created_at", "")).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if created < cutoff or created >= current_time:
                continue
            rec_mod = canonical_module(rec.get("module", ""), self._alias_map)
            if not target_mod or rec_mod != target_mod:
                continue
            ec_match = bool(error_code and (rec.get("error_code", "") or "").strip() == error_code)
            # Symptom: same error in subject/description, or similar subject
            rec_text = f"{rec.get('subject', '')} {rec.get('description', '')} {rec.get('error_code', '')}"
            symptom_match = False
            if error_code and error_code in rec_text:
                symptom_match = True
            if subject and _fuzzy_score(subject, rec.get("subject", "")) >= 0.65:
                symptom_match = True
            if ec_match or symptom_match:
                candidates.append(rec)
        candidates.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return candidates


# ══════════════════════════════════════════════════════════════════════════════
# Usage metrics index
# ══════════════════════════════════════════════════════════════════════════════

class UsageIndex:
    """Per-account usage trend computations."""

    def __init__(self, usage_df: pd.DataFrame):
        self._by_account: dict[str, list[dict]] = {}
        for _, row in usage_df.iterrows():
            rec = row.to_dict()
            acc_id = str(rec.get("account_id", "")).strip()
            if acc_id:
                self._by_account.setdefault(acc_id, []).append(rec)
        # Sort each account's records by month ascending
        for acc_id in self._by_account:
            self._by_account[acc_id].sort(key=lambda r: str(r.get("month", "")))

    def last_n_months(self, account_id: str, n: int = 3) -> list[dict]:
        return self._by_account.get(account_id.strip(), [])[-n:]

    def trend_summary(self, account_id: str, n: int = 3) -> dict:
        """Return a human-readable trend summary for the last n months."""
        rows = self.last_n_months(account_id, n)
        if not rows:
            return {"available": False, "summary": "No usage data available."}

        def _safe_float(v: Any) -> float | None:
            try:
                return float(v)
            except Exception:
                return None

        metrics = ["active_users", "login_rate_pct", "documents_processed", "tickets_opened"]
        result: dict[str, Any] = {"available": True, "months": [r.get("month") for r in rows]}

        for m in metrics:
            vals = [_safe_float(r.get(m)) for r in rows]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                delta = vals[-1] - vals[0]
                pct = (delta / vals[0] * 100) if vals[0] else 0.0
                direction = "stable"
                if pct <= -15:
                    direction = "declining"
                elif pct >= 15:
                    direction = "improving"
                result[m] = {
                    "values": vals,
                    "direction": direction,
                    "pct_change": round(pct, 1),
                    "latest": vals[-1],
                }
            elif vals:
                result[m] = {"values": vals, "direction": "stable", "pct_change": 0.0, "latest": vals[-1]}

        # Key signal: active_users decline > 25% → account risk override per KB-2914
        au = result.get("active_users", {})
        result["risk_override_usage"] = (
            isinstance(au, dict) and au.get("direction") == "declining" and au.get("pct_change", 0) <= -25
        )
        return result


# ══════════════════════════════════════════════════════════════════════════════
# KB search index (BM25)
# ══════════════════════════════════════════════════════════════════════════════

class KBIndex:
    """BM25-based knowledge-base search with metadata filtering."""

    def __init__(self, articles: list[dict], alias_map: dict[str, str] | None = None):
        """articles: list of {id, meta, body} from kb.load_kb()"""
        self._articles = articles
        self._by_id: dict[str, dict] = {a["id"]: a for a in articles}
        self._alias_map = alias_map or {}

        reverse_aliases: dict[str, list[str]] = {}
        for alias, canon in self._alias_map.items():
            reverse_aliases.setdefault(canon, []).append(alias)

        self._ids: list[str] = []
        corpus_tokens: list[list[str]] = []
        for art in articles:
            meta = art.get("meta", {})
            body_snippet = art.get("body", "")[:600]
            mod = _normalise(meta.get("module", ""))
            alias_text = " ".join(reverse_aliases.get(mod, []))
            text = " ".join([
                art["id"],
                meta.get("title", ""),
                meta.get("module", ""),
                alias_text,
                " ".join(meta.get("error_codes", []) if isinstance(meta.get("error_codes"), list) else []),
                meta.get("type", ""),
                body_snippet,
            ])
            tokens = _tokenise(text)
            self._ids.append(art["id"])
            corpus_tokens.append(tokens)

        self._bm25 = BM25Okapi(corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = 5,
        module_filter: str | None = None,
        type_filter: str | None = None,
        exclude_deprecated: bool = True,
    ) -> list[dict]:
        """Return top_k articles matching the query.

        Returns list of {id, title, module, type, status, snippet}.
        """
        tokens = _tokenise(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)

        # Build ranked list with filters
        ranked = []
        for idx, score in sorted(enumerate(scores), key=lambda x: -x[1]):
            art = self._articles[idx]
            meta = art.get("meta", {})
            status = meta.get("status", "current")

            if exclude_deprecated and status == "deprecated":
                continue
            if module_filter:
                art_mod = _normalise(meta.get("module", ""))
                if art_mod not in ("all", ""):
                    canon_filt = canonical_module(module_filter, self._alias_map)
                    canon_art = canonical_module(art_mod, self._alias_map)
                    matches = canon_filt == canon_art or canon_filt in art_mod or art_mod in canon_filt
                    if not matches:
                        continue
            if type_filter and meta.get("type", "") != type_filter:
                continue

            ranked.append({
                "id": art["id"],
                "title": meta.get("title", ""),
                "module": meta.get("module", ""),
                "type": meta.get("type", ""),
                "status": status,
                "error_codes": meta.get("error_codes", []),
                "snippet": art.get("body", "")[:200].strip(),
                "score": round(float(score), 3),
            })
            if len(ranked) >= top_k:
                break
        return ranked

    def get_article(self, article_id: str, follow_superseded: bool = True) -> dict | None:
        """Fetch a full article, following superseded_by chain if needed."""
        art = self._by_id.get(article_id)
        if not art:
            return None
        meta = art.get("meta", {})
        if follow_superseded and meta.get("status") == "deprecated":
            successor_id = meta.get("superseded_by", "")
            if successor_id and successor_id in self._by_id:
                successor = self._by_id[successor_id]
                return {**successor, "_followed_from": article_id}
        return art

    def get_policy_articles(self) -> list[dict]:
        """Return all policy-type articles (pre-filtered, always useful)."""
        return [a for a in self._articles if a.get("meta", {}).get("type") == "policy"
                and a.get("meta", {}).get("status") != "deprecated"]
