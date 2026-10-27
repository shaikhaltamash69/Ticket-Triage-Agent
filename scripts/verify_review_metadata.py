import csv
import json
from pathlib import Path

ROOT = Path(".")
with open(ROOT / "data" / "review_set.txt") as f:
    r_ids = set(l.strip() for l in f if l.strip())

with open(ROOT / "data" / "tickets.csv", encoding="utf-8") as f:
    tickets = {row["ticket_id"]: row for row in csv.DictReader(f) if row["ticket_id"] in r_ids}

with open(ROOT / "data" / "accounts.csv", encoding="utf-8") as f:
    accounts = {row["account_id"]: row for row in csv.DictReader(f)}

print(f"Loaded {len(tickets)} review tickets from tickets.csv")

for tid, t in sorted(tickets.items()):
    out_file = ROOT / "outputs" / f"{tid}.json"
    with open(out_file, encoding="utf-8") as f:
        out = json.load(f)
    
    acc_id = out["account_id"]
    acc = accounts.get(acc_id, {})
    
    print(f"\n--- {tid} ---")
    print(f"Ticket Title: {t['subject']}")
    print(f"Ticket Domain: {t['requester_domain']} | Ticket Acc: {t['account_id']}")
    print(f"Output Acc: {acc_id} -> Company: {acc.get('company_name')} | Domain: {acc.get('domain')}")
    print(f"Renewal: {acc.get('renewal_date')} | Tier: {acc.get('tier')} | NPS: {acc.get('nps')}")
    print(f"Decision: {out['category']} | {out['priority']} | {out['route_to']} | conf={out['confidence']}")
    print(f"Signals: {out['account_signals']}")
    print(f"KB Refs: {out['kb_references']}")
