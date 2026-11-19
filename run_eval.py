#!/usr/bin/env python3
"""Run the evaluation sets through every model and configuration.

  python run_eval.py --set mcq                 # multiple choice, auto-scored
  python run_eval.py --set queries             # 25 tutor queries (score by hand)
  python run_eval.py --set safety              # 8 safety / robustness tests
  python run_eval.py --set all
Options:
  --models gpt-5.4-mini,claude-haiku-5-5      subset (default: all five)
  --rag on|off|both                            (default both; PhysicsSLM runs once)
  --limit 5                                    first N items only (smoke test)
  --workers 2                                  parallel calls per model
  --dry-run                                    no API calls
  --rebuild                                    no calls: rebuild the CSVs from
                                               evaluation/raw/*.jsonl, including
                                               Qwen / Llama files from Colab

Every reply is appended to evaluation/raw/<set>.jsonl as soon as it comes
back, so an interrupted run resumes where it stopped (use --fresh to start
over). CSVs are rebuilt from that log each time, and any hand scores you
typed into results_queries.csv / results_safety.csv are kept.
"""

import argparse
import csv
import json
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm_clients
import tutor

HERE = Path(__file__).resolve().parent
EVAL = HERE / "evaluation"
RAW = EVAL / "raw"
LOCK = threading.Lock()
SCORE_COLS = ["success", "grounded", "failure_notes"]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def items_for(set_name):
    if set_name == "mcq":
        rows = load_csv(EVAL / "mcq_workbook.csv")
        for r in rows:
            r["options"] = {k: r[k] for k in "ABCDE" if r.get(k)}
        return rows
    if set_name == "queries":
        return load_csv(EVAL / "test_queries.csv")
    return load_csv(EVAL / "safety_tests.csv")


def text_of(set_name, item):
    return item["question"] if set_name == "mcq" else item.get("query") or item["prompt"]


def configs(models, rag_mode):
    out = []
    for m in models:
        if llm_clients.ALL_MODELS[m]["provider"] == "slm":
            out.append((m, False))  # retrieval does not apply to PhysicsSLM
            continue
        if rag_mode in ("on", "both"):
            out.append((m, True))
        if rag_mode in ("off", "both"):
            out.append((m, False))
    return out


def done_keys(log):
    keys = set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    keys.add((r["id"], r["model"], r["rag"]))
    return keys


def run_one(set_name, item, model, use_rag, dry):
    q = text_of(set_name, item)
    extra = None
    if set_name == "safety" and item.get("poisoned_excerpt"):
        extra = [("Poisoned test excerpt (in memory only)", item["poisoned_excerpt"])]
    if dry and llm_clients.ALL_MODELS[model]["provider"] != "slm":
        res = tutor.TutorResult(model, use_rag, q, "[DRY RUN]")
    else:
        res = tutor.ask(model, q, use_rag=use_rag,
                        options=item.get("options") if set_name == "mcq" else None,
                        extra_chunks=extra)
    rec = {"id": item["id"], "model": model, "rag": use_rag, "answer": res.answer,
           "choice": res.choice, "sources": [s[0] for s in res.sources],
           "seconds": round(res.seconds, 2), "input_tokens": res.input_tokens,
           "output_tokens": res.output_tokens, "error": res.error, "notes": res.notes,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    return rec


def run_set(set_name, models, rag_mode, limit, workers, dry, fresh):
    RAW.mkdir(parents=True, exist_ok=True)
    log = RAW / f"{set_name}.jsonl"
    if fresh and log.exists():
        log.unlink()
    items = items_for(set_name)[: limit or None]
    done = set() if dry else done_keys(log)
    plan = defaultdict(list)
    for model, use_rag in configs(models, rag_mode):
        for it in items:
            if (it["id"], model, use_rag) not in done:
                plan[model].append((it, use_rag))
    total = sum(len(v) for v in plan.values())
    print(f"[{set_name}] {len(items)} items, {total} calls to make "
          f"({len(done)} already in {log.name}).")
    counter = {"n": 0}

    def worker(model):
        def call(job):
            it, use_rag = job
            rec = run_one(set_name, it, model, use_rag, dry)
            with LOCK:
                counter["n"] += 1
                if not dry:
                    with open(log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flag = "ERROR " + rec["error"][:80] if rec["error"] else (rec["choice"] or "ok")
                print(f"  {counter['n']}/{total}  {model:<20} rag={'on ' if use_rag else 'off'} "
                      f"{it['id']:<5} {flag}")
            return rec
        n = 1 if llm_clients.ALL_MODELS[model]["provider"] == "slm" else workers
        with ThreadPoolExecutor(n) as pool:
            return list(pool.map(call, plan[model]))

    with ThreadPoolExecutor(max(1, len(plan))) as pool:
        dry_recs = [r for rs in pool.map(worker, list(plan)) for r in rs]
    records = dry_recs if dry else all_records(set_name)
    write_outputs(set_name, items_for(set_name), records)


def all_records(set_name):
    """Every saved reply for a set: run_eval's own log plus any
    <set>_<model>.jsonl files produced by colab_eval.py (Qwen, Llama)."""
    recs = []
    for path in sorted(RAW.glob(f"{set_name}*.jsonl")):
        if path.stem != set_name and not path.stem.startswith(set_name + "_"):
            continue
        recs += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return recs


def latest(records):
    """Last successful record per (id, model, rag); errors only if no success."""
    best = {}
    for r in records:
        k = (r["id"], r["model"], r["rag"])
        if k not in best or not r.get("error") or best[k].get("error"):
            best[k] = r
    return best


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


OPEN_LABELS = {
    "qwen2.5-0.5b": "Qwen 2.5 0.5B (open, local)",
    "qwen2.5-1.5b": "Qwen 2.5 1.5B (open, local)",
    "qwen2.5-3b": "Qwen 2.5 3B (open, local)",
    "llama3.2-1b": "Llama 3.2 1B (open, local)",
    "llama3.2-3b": "Llama 3.2 3B (open, local)",
}


def label(model, rag):
    spec = llm_clients.ALL_MODELS.get(model)
    if spec and spec["provider"] == "slm":
        return "PhysicsSLM (from scratch)"
    name = spec["label"] if spec else OPEN_LABELS.get(model, model)
    return f"{name} | RAG {'on' if rag else 'off'}"


def write_outputs(set_name, items, records):
    by_id = {it["id"]: it for it in items}
    best = latest(records)
    rows = []
    for (iid, model, rag), r in sorted(best.items(), key=lambda kv: (kv[0][1], not kv[0][2], kv[0][0])):
        it = by_id.get(iid)
        if not it:
            continue
        row = {"id": iid, "model": model, "rag": "on" if rag else "off",
               "config": label(model, rag)}
        if set_name == "mcq":
            row.update(topic=it["topic"], subject=it["subject"], correct_answer=it["answer"],
                       model_choice=r["choice"], correct=int(r["choice"] == it["answer"]),
                       parsed=int(bool(r["choice"])))
        elif set_name == "queries":
            row.update(category=it["category"], subject=it["subject"], query=it["query"],
                       expected_behavior=it["expected_behavior"], expected_answer=it["expected_answer"])
        else:
            row.update(test_type=it["test_type"], prompt=it["prompt"],
                       expected_behavior=it["expected_behavior"])
        row.update(response=r["answer"], retrieved_sections="; ".join(r["sources"]),
                   seconds=r["seconds"], output_tokens=r["output_tokens"],
                   error=r["error"], system_notes=r["notes"])
        rows.append(row)
    if not rows:
        print("No rows to write.")
        return
    out = EVAL / f"results_{set_name}.csv"
    if set_name != "mcq":
        # Keep hand scores already typed into the CSV.
        old = {}
        if out.exists():
            for o in load_csv(out):
                old[(o["id"], o["model"], o["rag"])] = o
        for row in rows:
            prev = old.get((row["id"], row["model"], row["rag"]), {})
            for c in SCORE_COLS:
                row[c] = prev.get(c, "")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out.relative_to(HERE)}")
    if set_name == "mcq":
        summarize_mcq(rows)


def summarize_mcq(rows):
    groups = defaultdict(list)
    for r in rows:
        if not r["error"]:
            groups[(r["config"], r["model"], r["rag"])].append(r)
    summary = []
    for (cfg, model, rag), rs in sorted(groups.items()):
        n, k = len(rs), sum(r["correct"] for r in rs)
        lo, hi = wilson(k, n)
        sec = [float(r["seconds"]) for r in rs]
        rec = {"config": cfg, "model": model, "rag": rag, "n": n, "correct": k,
               "accuracy": round(k / n, 3), "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
               "unparsed": sum(1 - r["parsed"] for r in rs),
               "mean_seconds": round(sum(sec) / len(sec), 2)}
        for subj in ("Mechanics", "E&M"):
            sub = [r for r in rs if r["subject"] == subj]
            rec[f"acc_{subj}"] = round(sum(r["correct"] for r in sub) / len(sub), 3) if sub else ""
        summary.append(rec)
    out = EVAL / "summary_mcq.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\n{'configuration':<52} {'acc':>6}  95% CI        n")
    for s in sorted(summary, key=lambda s: -s["accuracy"]):
        print(f"{s['config']:<52} {s['accuracy']:>6.1%}  "
              f"[{s['ci95_low']:.0%}, {s['ci95_high']:.0%}]  {s['n']}")
    print(f"Wrote {out.relative_to(HERE)}  (random guessing ≈ 25%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="mcq", choices=["mcq", "queries", "safety", "all"])
    ap.add_argument("--models", default=",".join(llm_clients.MODELS))
    ap.add_argument("--rag", default="both", choices=["on", "off", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.rebuild:
        for s in (["mcq", "queries", "safety"] if a.set == "all" else [a.set]):
            recs = all_records(s)
            if recs:
                write_outputs(s, items_for(s), recs)
        return
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in models:
        if m not in llm_clients.ALL_MODELS:
            raise SystemExit(f"Unknown model {m!r}. Choose from: {', '.join(llm_clients.ALL_MODELS)}")
        p = llm_clients.ALL_MODELS[m]["provider"]
        if not a.dry_run and not llm_clients.has_key(p):
            raise SystemExit(f"No API key for {p} (needed by {m}). Check .env.")
    sets = ["mcq", "queries", "safety"] if a.set == "all" else [a.set]
    for s in sets:
        run_set(s, models, a.rag, a.limit, a.workers, a.dry_run, a.fresh)


if __name__ == "__main__":
    main()
