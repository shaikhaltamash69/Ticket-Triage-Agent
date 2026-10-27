"""Run the triage agent.

  python run.py --set review            # the 10 review tickets (required)
  python run.py --set sample --n 50     # random sample (optional)
  python run.py --ids TKT-123456 TKT-234567

Writes one JSON per ticket to outputs/ plus outputs/run_log.jsonl.
"""
import argparse, json, random, sys, time, traceback
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tam_agent.data import load_accounts, load_tickets, load_usage, load_review_set
from tam_agent.kb import load_kb
from tam_agent.agent import Context, TAMAgent
from tam_agent.schema import validate

OUT = Path(__file__).resolve().parent / "outputs"
REPORTS = Path(__file__).resolve().parent / "verification"


def is_fallback(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return is_fallback_data(data)
    except Exception:
        return True


def is_fallback_data(data: dict) -> bool:
    return (
        "Emergency fallback" in str(data.get("internal_note", ""))
        or data.get("confidence", 0) <= 0.1
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["review", "sample"], default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--force", action="store_true", help="Re-run tickets even if valid output exists")
    args = ap.parse_args()

    tickets = load_tickets()
    ctx = Context(load_accounts(), tickets, load_usage(), load_kb())
    if args.ids:
        ids = args.ids
    elif args.set == "sample":
        ids = random.Random(args.seed).sample(list(tickets["ticket_id"]), args.n)
    else:
        ids = load_review_set()

    OUT.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    by_id = tickets.set_index("ticket_id")
    ok = 0
    with open(OUT / "run_log.jsonl", "a", encoding="utf-8") as log:
        for tid in ids:
            p = OUT / f"{tid}.json"
            if p.exists() and not is_fallback(p) and not args.force and not args.ids:
                print(f"{tid}: skipping (valid non-fallback output already exists)")
                ok += 1
                continue

            t0, calls0 = time.time(), ctx.llm.calls
            rec = {"ticket_id": tid}
            try:
                ticket = by_id.loc[tid].to_dict() | {"ticket_id": tid}
                agent = TAMAgent(ticket, ctx)
                result = agent.run()
                errs = validate(result)
                # A provider/network failure is schema-shaped but never a usable
                # review result.  Do not let a retry replace an existing genuine
                # agent output with that fallback.
                if is_fallback_data(result):
                    rec |= {"status": "fallback", "errors": ["LLM provider did not produce a usable decision"]}
                    print(f"{tid}: fallback retained; output was not overwritten")
                else:
                    (OUT / f"{tid}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                    # This audit record is produced from the same live agent run. It
                    # makes reviewable the evidence intentionally omitted from the
                    # fixed submission schema (notably source sentences for reply steps).
                    (REPORTS / f"{tid}.json").write_text(
                        json.dumps(agent.verification, indent=2), encoding="utf-8"
                    )
                    rec |= {"status": "ok" if not errs else "schema_error", "errors": errs}
                    ok += not errs
            except Exception as e:
                rec |= {"status": "exception", "errors": [repr(e)], "traceback": traceback.format_exc(limit=3)}
            rec |= {"seconds": round(time.time() - t0, 2), "llm_calls": ctx.llm.calls - calls0}
            log.write(json.dumps(rec) + "\n")
            print(f"{tid}: {rec['status']} ({rec['seconds']}s, {rec['llm_calls']} LLM calls)")
            time.sleep(15)  # lets the TPM window recover between tickets
    print(f"\n{ok}/{len(ids)} tickets produced valid output")


if __name__ == "__main__":
    main()
