import glob
import json
import os
import re

output_files = sorted(glob.glob('outputs/TKT-*.json'))
print(f"Found {len(output_files)} files in outputs/")

for path in output_files:
    tkt = os.path.basename(path).replace('.json', '')
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f"\n==================== {tkt} ====================")
    print(f"Category: {d.get('category')} | Priority: {d.get('priority')} | Route: {d.get('route_to')} | Conf: {d.get('confidence')}")
    print(f"Account ID: {d.get('account_id')}")
    print(f"Account Signals: {d.get('account_signals')}")
    print(f"KB References: {d.get('kb_references')}")
    print(f"Steps count: {len(d.get('reasoning_trace', []))}")
    note = d.get('internal_note', '')
    reply = d.get('customer_reply', '')
    
    # Check for any URLs in customer reply or internal note
    urls_reply = re.findall(r'https?://[^\s)\]]+', reply)
    urls_note = re.findall(r'https?://[^\s)\]]+', note)
    if urls_reply:
        print(f"  [URL ALERT in reply]: {urls_reply}")
    if urls_note:
        print(f"  [URL ALERT in note]: {urls_note}")
        
    # Check for fallback signals
    if d.get('confidence', 0) < 0.6:
        print(f"  [LOW CONFIDENCE]: {d.get('confidence')}")
    if 'fallback' in note.lower() or 'default' in note.lower() or 'limits' in note.lower() or 'error' in note.lower():
        print(f"  [NOTE KEYWORD WARN]: {note[:160]}")
    else:
        print(f"  Note snippet: {note[:120]}...")
    print(f"  Reply snippet: {reply[:120]}...")
