"""Quick smoke test for a single ticket before running the full review set."""
import sys, json, traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("Loading data...", flush=True)
from tam_agent.data import load_accounts, load_tickets, load_usage
from tam_agent.kb import load_kb
from tam_agent.agent import Context, triage
from tam_agent.schema import validate

accounts = load_accounts()
tickets  = load_tickets()
usage    = load_usage()
kb       = load_kb()

print(f"  Loaded: {len(accounts)} accounts, {len(tickets)} tickets, {len(usage)} usage rows, {len(kb)} KB articles", flush=True)

ctx = Context(accounts, tickets, usage, kb)

# Test single review ticket
by_id = tickets.set_index("ticket_id")
tid   = "TKT-144939"
ticket = by_id.loc[tid].to_dict()
ticket["ticket_id"] = tid

print(f"\nTicket {tid}:", flush=True)
print(f"  Subject:  {ticket.get('subject')}", flush=True)
print(f"  Module:   {ticket.get('module')}", flush=True)
print(f"  Domain:   {ticket.get('requester_domain')}", flush=True)
print(f"  AcctID:   {ticket.get('account_id')}", flush=True)
print(f"\nRunning triage...", flush=True)

try:
    result = triage(ticket, ctx)
    errs   = validate(result)
    print(f"\n{'='*60}", flush=True)
    print(f"RESULT:", flush=True)
    print(f"  ticket_id:   {result.get('ticket_id')}", flush=True)
    print(f"  account_id:  {result.get('account_id')}", flush=True)
    print(f"  category:    {result.get('category')}", flush=True)
    print(f"  priority:    {result.get('priority')}", flush=True)
    print(f"  route_to:    {result.get('route_to')}", flush=True)
    print(f"  escalate:    {result.get('escalate')}", flush=True)
    print(f"  duplicate:   {result.get('duplicate_of')}", flush=True)
    print(f"  kb_refs:     {result.get('kb_references')}", flush=True)
    print(f"  signals:     {result.get('account_signals')}", flush=True)
    print(f"  steps:       {len(result.get('reasoning_trace', []))}", flush=True)
    print(f"  confidence:  {result.get('confidence')}", flush=True)
    print(f"\nSchema errors: {errs}", flush=True)
    print(f"\nCustomer reply:\n{result.get('customer_reply', '')}", flush=True)
    print(f"\nInternal note:\n{result.get('internal_note', '')}", flush=True)
    if errs:
        sys.exit(1)
    else:
        print("\n[PASS] Valid output!", flush=True)
except Exception as e:
    print(f"\n[FAIL] Exception: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
