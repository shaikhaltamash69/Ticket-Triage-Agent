import json
import os
import re
from pathlib import Path

ROOT = Path(".")
REVIEW_FILE = ROOT / "data" / "review_set.txt"
OUTPUTS_DIR = ROOT / "outputs"
ACCOUNTS_FILE = ROOT / "data" / "accounts.csv"
KB_DIR = ROOT / "kb"

with open(REVIEW_FILE, "r", encoding="utf-8") as f:
    review_ids = [line.strip() for line in f if line.strip()]

import csv
with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    valid_acc_ids = {row["account_id"] for row in reader}

kb_files = set(os.listdir(KB_DIR))
valid_kb_ids = {f.replace(".md", "") for f in kb_files if f.endswith(".md")}

print(f"Loaded {len(review_ids)} review tickets, {len(valid_acc_ids)} accounts, {len(valid_kb_ids)} KB articles.")

url_pattern = re.compile(r'https?://|www\.|\.com/|\.org/|\[.*?\]\(.*?\)')
fallback_keywords = [
    "emergency_fallback", "fallback", "timed out", "timeout",
    "connection error", "processing limits", "defaults due to"
]

all_clean = True

for tid in review_ids:
    out_path = OUTPUTS_DIR / f"{tid}.json"
    if not out_path.exists():
        print(f"[FAIL] Missing {out_path}")
        all_clean = False
        continue

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. URL scan
    text_to_check = json.dumps({
        "customer_reply": data.get("customer_reply", ""),
        "internal_note": data.get("internal_note", ""),
    })
    urls_found = url_pattern.findall(text_to_check)
    if urls_found:
        print(f"[WARN URL] {tid} has URLs or markdown links: {urls_found}")
        all_clean = False

    # 2. Fallback scan
    for kw in fallback_keywords:
        if kw in text_to_check.lower():
            print(f"[WARN FALLBACK] {tid} contains keyword '{kw}'")
            all_clean = False

    # 3. Confidence scan
    conf = data.get("confidence", 0)
    if conf < 0.6:
        print(f"[WARN CONFIDENCE] {tid} low confidence: {conf}")
        all_clean = False

    # 4. Account ID check
    acc_id = data.get("account_id")
    if not acc_id or acc_id not in valid_acc_ids:
        print(f"[WARN ACCOUNT] {tid} invalid or missing account_id: {acc_id}")
        all_clean = False

    # 5. KB references check
    kb_refs = data.get("kb_references", [])
    for ref in kb_refs:
        if ref not in valid_kb_ids:
            print(f"[WARN KB] {tid} invalid kb_reference: {ref}")
            all_clean = False

    # 6. Reasoning trace
    trace = data.get("reasoning_trace", [])
    if len(trace) < 3:
        print(f"[WARN TRACE] {tid} trace length too short: {len(trace)}")
        all_clean = False

    print(f"[CHECK] {tid}: category={data.get('category')}, priority={data.get('priority')}, route_to={data.get('route_to')}, acc={acc_id}, conf={conf}, steps={len(trace)}, kbs={kb_refs}")

if all_clean:
    print("\n>>> ALL 10 TICKETS PASSED URL SCAN, FALLBACK SCAN, ACCOUNT CHECK, AND KB CHECK! <<<")
else:
    print("\n>>> WARNING: ISSUES FOUND IN OUTPUTS! <<<")
