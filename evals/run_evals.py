"""Run multi-turn scenarios against the real pipeline and score them.

    # against a real model (reads LLM_* env vars, see .env.example)
    python evals/run_evals.py --repeats 3 --out runs/candidate.json

    # plumbing check with the rule-based demo model (what CI runs)
    python evals/run_evals.py --demo --tag smoke --min-pass-rate 1.0

Each repeat is a fresh conversation through AgentService: same store, tools, guards and
handoff logic as production, only the channel is skipped. Results are written as JSON so
two runs can be compared case by case (see github.com/123idog/llm-eval-kit).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_service.app import build_llm  # noqa: E402
from agent_service.catalog import Catalog  # noqa: E402
from agent_service.config import Settings  # noqa: E402
from agent_service.llm import DemoLLM  # noqa: E402
from agent_service.pipeline import AgentService, Inbound  # noqa: E402
from agent_service.store import MemoryStore  # noqa: E402

# Fixed "today" so date-dependent scenarios (Saturday, "this Thursday") are reproducible.
NOW = datetime(2026, 6, 2, 10, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))


def check(expect: dict, outcomes: list, store: MemoryStore, conv_id: str) -> list[str]:
    final = outcomes[-1]
    text = final.reply or ""
    all_tools = [t for o in outcomes for t in o.tool_calls]
    failures = []
    if "language" in expect and final.language != expect["language"]:
        failures.append(f"language {final.language} != {expect['language']}")
    if "contains_any" in expect and not any(s in text for s in expect["contains_any"]):
        failures.append(f"reply has none of {expect['contains_any']}")
    for s in expect.get("not_contains", []):
        if any(s.lower() in (o.reply or "").lower() for o in outcomes):
            failures.append(f"a reply contains {s!r}")
    for t in expect.get("tools_include", []):
        if t not in all_tools:
            failures.append(f"tool {t} not called")
    if "handoff" in expect and any(o.handed_off for o in outcomes) != expect["handoff"]:
        failures.append(f"handoff expected {expect['handoff']}")
    if "lead" in expect and (store.get_lead(conv_id) is not None) != expect["lead"]:
        failures.append(f"lead expected {expect['lead']}")
    for k in expect.get("facts_include", []):
        if k not in store.get_facts(conv_id):
            failures.append(f"fact {k} not saved")
    if any(
        o.status in ("fallback_handoff", "llm_error_handoff") for o in outcomes
    ) and not expect.get("handoff"):
        failures.append("fell back to a safe message (guards or model failed)")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(Path(__file__).with_name("scenarios.jsonl")))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tag", help="only scenarios with this tag")
    ap.add_argument("--demo", action="store_true", help="use the rule-based demo model")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--min-pass-rate", type=float, default=0.0)
    args = ap.parse_args()

    settings = Settings.from_env()
    llm = DemoLLM() if args.demo else build_llm(settings)
    catalog = Catalog.load(settings.catalog_path or None)
    scenarios = [
        json.loads(line)
        for line in Path(args.scenarios).read_text("utf-8").splitlines()
        if line.strip()
    ]
    if args.tag:
        scenarios = [s for s in scenarios if args.tag in s.get("tags", [])]

    results = []
    for sc in scenarios:
        for r in range(args.repeats):
            store = MemoryStore()
            service = AgentService(store, llm, catalog, settings, clock=lambda: NOW)
            conv_id = f"eval-{sc['id']}-{uuid.uuid4().hex[:6]}"
            outcomes = [
                service.handle(
                    Inbound(message_id=f"{conv_id}-{i}", conversation_id=conv_id, text=t)
                )
                for i, t in enumerate(sc["turns"])
            ]
            failures = check(sc.get("expect", {}), outcomes, store, conv_id)
            turns = list(store.turns(conv_id))
            results.append(
                {
                    "id": sc["id"],
                    "repeat": r,
                    "passed": not failures,
                    "failures": failures,
                    "replies": [o.reply for o in outcomes],
                    "tools": [o.tool_calls for o in outcomes],
                    "input_tokens": sum(t.input_tokens for t in turns),
                    "output_tokens": sum(t.output_tokens for t in turns),
                    "cost_usd": round(sum(t.cost_usd for t in turns), 6),
                    "guard_reasons": [t.guard_reasons for t in turns],
                }
            )

    by_case: dict[str, list[dict]] = {}
    for res in results:
        by_case.setdefault(res["id"], []).append(res)
    print(
        f"model: {'demo' if args.demo else settings.llm_model}  scenarios: {len(by_case)}  repeats: {args.repeats}\n"
    )
    print("| Case | Pass | Cost (USD) | First failure |")
    print("|---|---|---|---|")
    for cid, rs in by_case.items():
        passed = sum(r["passed"] for r in rs)
        first = next((r["failures"][0] for r in rs if r["failures"]), "")
        cost = sum(r["cost_usd"] for r in rs) / len(rs)
        print(f"| {cid} | {passed}/{len(rs)} | {cost:.5f} | {first} |")
    rate = sum(r["passed"] for r in results) / max(len(results), 1)
    print(f"\npass rate: {rate:.0%}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {"model": "demo" if args.demo else settings.llm_model, "results": results},
                ensure_ascii=False,
                indent=2,
            ),
            "utf-8",
        )
    return 0 if rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
