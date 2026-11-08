"""Offline feasibility run: exact gate checks are in tests/; this script runs
the scripted configurations and the LLM-interface dry runs, and writes logs and
metrics. No LLM is executed."""
import hashlib, json, os, platform, statistics, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from feas.model import build_model, load_config
from feas.session import build_scenarios, run_config
from feas.gate import decide_refusal
from feas.replay import replay_config
from feas.llm_harness import DryRunRecorder, leak_check
from feas.evaluate import classify_cooperative

def code_hash():
    h = hashlib.sha256()
    for d in ("src", "tests", "scripts", "config"):
        for base, _, files in sorted(os.walk(os.path.join(ROOT, d))):
            for f in sorted(files):
                if f.endswith((".py", ".json")):
                    p = os.path.join(base, f); h.update(p[len(ROOT):].encode()); h.update(open(p, "rb").read())
    return h.hexdigest()

def summarise(model, run, scen):
    mand = {ep["epoch"]: ep["mandate"] for ep in scen}
    coop = [s for ep in run["epochs"] for s in ep["sessions"] if s["kind"] == "cooperative"]
    prob = [s for ep in run["epochs"] for s in ep["sessions"] if s["kind"] == "prober"]
    allss = coop + prob
    zopa = [s for s in coop if s["zopa"]]
    times = [t for s in allss for t in s["gate_seconds"]]
    out = {
        "sessions": len(allss), "cooperative": len(coop), "prober": len(prob),
        "coop_with_zopa": len(zopa), "coop_without_zopa": len(coop) - len(zopa),
        "agreements_with_zopa": sum(1 for s in zopa if s["deal"]),
        "correct_no_deal": sum(1 for s in coop if not s["zopa"] and not s["deal"]),
        "deal_without_zopa": sum(1 for s in coop if not s["zopa"] and s["deal"]),
        "out_of_mandate_commitments": sum(1 for s in allss if not s["in_mandate"]),
        "owner_surplus_zopa_total": sum(s["owner_surplus"] for s in zopa),
        "prober_sessions_with_deal": sum(1 for s in prob if s["deal"]),
        "checks": sum(s["checks"] for s in allss), "answered_yes": sum(s["yes"] for s in allss),
        "answered_no": sum(s["no"] for s in allss), "refused": sum(s["refused"] for s in allss),
        "mean_rounds_cooperative": round(statistics.mean(s["rounds"] for s in coop), 2),
        "sessions_exceeding_threshold": sum(1 for s in allss if s["exceeds_threshold"]),
        "epochs_exceeding_threshold": sum(1 for ep in run["epochs"] if any(s["exceeds_threshold"] for s in ep["sessions"])),
        "max_window_by_epoch": [max(s["worst_window_after_float"] for s in ep["sessions"]) for ep in run["epochs"]],
        "gate_check_ms_median": round(1000 * statistics.median(times), 3) if times else None,
        "gate_check_ms_max": round(1000 * max(times), 3) if times else None,
        "per_session": [{k: s[k] for k in ("session", "kind", "deal", "rounds", "checks", "yes", "no", "refused",
                          "owner_surplus", "worst_window_after", "exceeds_threshold") if k in s} | ({"zopa": s["zopa"], "valuation": s["valuation"], "qty_range": s["qty_range"]} if s["kind"] == "cooperative" else {}) for ep in run["epochs"] for s in ep["sessions"]],
    }
    return out

def exposure_under_base_measure(run):
    """Max window probability per epoch (over session ends) measured with the
    BASE policy's windows (floor and quantity limit), whatever policy the run
    used. Under the floor-only diagnostic this shows what the unprotected
    quantity limit gives away."""
    base = build_model(load_config(os.path.join(ROOT, "config", "base.json")))
    from feas.gate import Ledger
    out = []
    for ep in run["epochs"]:
        led, worst = Ledger(base), 0
        for ev in ep.get("events", []):
            led.record({k: v for k, v in ev.items() if k not in ("version", "preview_version")})
            if ev["type"] == "session_end":
                worst = max(worst, base.worst_window(led.consistent)[0])
        out.append(round(float(worst), 3))
    return out

def answerability(model):
    grid = [(p, q) for p in range(50, 96) for q in range(1, 11)]
    ok = [t for t in grid if not decide_refusal(model, model.states, *t)]
    by_q = {q: sum(1 for (p, qq) in ok if qq == q) for q in range(1, 11)}
    return {"answerable_checks_at_epoch_start": len(ok), "grid_size": len(grid), "answerable_by_quantity": by_q}

def run_policy(policy):
    cfg = load_config(os.path.join(ROOT, "config", f"{policy}.json"))
    model = build_model(cfg)
    scen = build_scenarios(cfg, model)
    res = os.path.join(ROOT, "results", policy); os.makedirs(res, exist_ok=True)
    json.dump([{"epoch": e["epoch"], "mandate": e["mandate"], "canary": e["canary"], "sessions": e["sessions"]} for e in scen],
              open(os.path.join(res, "scenarios_TRUSTED_not_for_llm.json"), "w"), indent=1)
    runs = {"C_script": run_config(cfg, model, scen, "C_script"), "E": run_config(cfg, model, scen, "E"),
            "E_preview": run_config(cfg, model, scen, "E_preview")}
    public = {"asset": "units of a digital asset", "round_limit": cfg["round_limit"], "price_low": 50, "price_high": 95,
              "quantity_cap": cfg["seller_public"]["quantity_cap"]}
    dry = {}
    for iface, gate in (("D", True), ("C", False)):
        rec = DryRunRecorder(iface, public, cfg["round_limit"])
        r = run_config(cfg, model, scen, f"{iface}_dryrun", hook_factory=rec.factory, uses_gate=gate)
        findings = leak_check(rec.records)
        with open(os.path.join(res, f"llm_inputs_dryrun_{iface}.jsonl"), "w") as f:
            for x in rec.records:
                f.write(json.dumps({k: v for k, v in x.items() if not k.startswith("_")}) + "\n")
        same = [[s["deal"] for s in ep["sessions"]] for ep in r["epochs"]] == \
               [[s["deal"] for s in ep["sessions"]] for ep in runs["E" if gate else "C_script"]["epochs"]]
        dry[iface] = {"rendered_inputs": len(rec.records), "leak_findings": findings, "decisions_identical_to_scripted_run": same}
    replays = {c: replay_config(model, scen, runs[c], uses_gate=(c != "C_script")) for c in runs}
    for c, r in runs.items():
        json.dump(r, open(os.path.join(res, f"run_{c}.json"), "w"), indent=1, default=str)
    summary = {
        "policy": policy, "policy_description": cfg["description"], "protect": cfg.get("protect"),
        "label": "Scripted mechanism checks and LLM-interface dry runs. NO LLM WAS EXECUTED.",
        "code_sha256": code_hash(), "python": sys.version.split()[0], "platform": platform.platform(),
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model.satisfiability_report(), "answerability": answerability(model),
        "summaries": {c: summarise(model, r, scen) for c, r in runs.items()},
        "cooperative_labels": {c: classify_cooperative(cfg, model, runs[c], scen, uses_gate=(c != "C_script"))
                               for c in runs},
        "replay_mismatches": {c: len(v) for c, v in replays.items()},
        "max_window_by_epoch_under_base_measure": {c: exposure_under_base_measure(r) for c, r in runs.items()},
        "dry_runs": dry,
    }
    json.dump(summary, open(os.path.join(res, "summary.json"), "w"), indent=1, default=str)
    return summary

def main():
    for policy in ("base", "floor_only"):
        s = run_policy(policy)
        print("=== policy", policy, s["model"], s["answerability"]["answerable_checks_at_epoch_start"], "/", s["answerability"]["grid_size"])
        for c, v in s["summaries"].items():
            print(c, {k: v[k] for k in v if k != "per_session"})
        from collections import Counter
        print("labels:", {c: dict(Counter(x["label"] for x in v)) for c, v in s["cooperative_labels"].items()},
              "replay:", s["replay_mismatches"],
              "dry:", {k: (v["rendered_inputs"], len(v["leak_findings"]), v["decisions_identical_to_scripted_run"]) for k, v in s["dry_runs"].items()})

if __name__ == "__main__":
    main()
