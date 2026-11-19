"""Train PhysicsSLM on a curriculum and score it on hidden probes.

Used by the "LLMs teach the from-scratch model" experiment. It does not edit
mathematics.py, tensor.py, Transformer.py, or dataloader.py, and it never
imports dataloader.py (importing it starts a training run). The model is
the Transformer class from Transformer.py; training is train_steps() from
physics_slm_core.py.

Every curriculum is mapped onto the SAME fixed vocabulary (teach/vocab.txt),
so models trained on different curricula have identical shapes and can be
compared on the same probes. A sentence that uses a word outside the
vocabulary is dropped whole, and the count is reported.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

TEACH = Path(__file__).resolve().parent
ROOT = TEACH.parent
sys.path.insert(0, str(ROOT))

from physics_slm_core import count_parameters, train_steps  # noqa: E402
from Transformer import Transformer  # noqa: E402

EOS = "<eos>"
DEFAULT_CONFIG = {"n_embed": 32, "num_heads": 4, "n_blocks": 3, "block_size": 16,
                  "lr": 0.05, "seed": 0}


def load_vocab(path=TEACH / "vocab.txt"):
    words = [w.strip() for w in Path(path).read_text(encoding="utf-8").split() if w.strip()]
    return {w: i for i, w in enumerate(words)}


def sentences_of(text):
    """Split a curriculum into sentences on <eos> (or newlines)."""
    text = text.replace("\r", "")
    if EOS in text:
        parts = text.split(EOS)
    else:
        parts = text.split("\n")
    return [" ".join(p.lower().split()) for p in parts if p.strip()]


def encode_curriculum(text, stoi, max_tokens=None):
    """Token ids for the in-vocabulary sentences, each followed by <eos>."""
    kept, dropped, ids = [], [], []
    for s in sentences_of(text):
        words = s.split()
        if all(w in stoi for w in words):
            if max_tokens and len(ids) + len(words) + 1 > max_tokens:
                break
            ids.extend(stoi[w] for w in words)
            ids.append(stoi[EOS])
            kept.append(s)
        else:
            dropped.append(s)
    return ids, kept, dropped


def make_state(ids, stoi, config):
    """The fields train_steps() reads, built for a fixed vocabulary.

    Mirrors physics_slm_core.initial_state, except the vocabulary comes from
    vocab.txt instead of from the corpus itself.
    """
    np.random.seed(int(config["seed"]))
    model = Transformer(len(stoi), int(config["n_embed"]), int(config["num_heads"]),
                        int(config["block_size"]), int(config["n_blocks"]))
    return {
        "model": model, "stoi": stoi, "itos": {i: w for w, i in stoi.items()},
        "ids": ids, "vocab_size": len(stoi), "block_size": int(config["block_size"]),
        "lr": float(config["lr"]), "losses": [], "hits": 0, "seen": 0,
        "steps_done": 0, "target": 0, "elapsed": 0.0,
        "rng": np.random.get_state(), "parameters": count_parameters(model),
    }


def load_probes(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["options"] = r["options"].split("|")
    return rows


def _logprob(state, context, target):
    """Sum log P(target words | context) using the model's own forward()."""
    model, block, stoi = state["model"], state["block_size"], state["stoi"]
    ids = [stoi[EOS]] + [stoi[w] for w in context.split()]
    total = 0.0
    for w in target.split():
        window = ids[-block:]
        row = model.forward(window)[len(window) - 1].to_list()
        total += math.log(max(float(row[stoi[w]]), 1e-12))
        ids.append(stoi[w])
    return total


def score_probes(state, probes):
    """Choice accuracy (most likely option, mean log-prob per word) and,
    for one-word answers, top-1 accuracy over the whole vocabulary."""
    stoi, block = state["stoi"], state["block_size"]
    per = []
    for p in probes:
        scores = {o: _logprob(state, p["prefix"], o) / len(o.split()) for o in p["options"]}
        pick = max(scores, key=scores.get)
        top1 = None
        if len(p["answer"].split()) == 1:
            ids = ([stoi[EOS]] + [stoi[w] for w in p["prefix"].split()])[-block:]
            row = state["model"].forward(ids)[len(ids) - 1].to_list()
            top1 = state["itos"][int(np.argmax(row))] == p["answer"]
        per.append({"id": p["id"], "kind": p["kind"], "pick": pick,
                    "correct": pick == p["answer"], "top1": top1,
                    "chance": 1 / len(p["options"])})
    return per


def summarize(per):
    out = {"choice_acc": sum(r["correct"] for r in per) / len(per),
           "chance": sum(r["chance"] for r in per) / len(per)}
    t = [r["top1"] for r in per if r["top1"] is not None]
    out["top1_acc"] = sum(t) / len(t) if t else None
    for kind in sorted({r["kind"] for r in per}):
        sub = [r for r in per if r["kind"] == kind]
        out[f"acc_{kind}"] = sum(r["correct"] for r in sub) / len(sub)
    return out


def run(name, text, steps=3000, checkpoints=(0, 500, 1000, 2000, 3000), seed=0,
        config=None, probes_path=TEACH / "probes_hidden.csv", max_tokens=15000,
        out_dir=TEACH / "results"):
    """Train one student on one curriculum and write results/<name>_seed<k>.json."""
    config = {**DEFAULT_CONFIG, **(config or {}), "seed": seed}
    stoi = load_vocab()
    ids, kept, dropped = encode_curriculum(text, stoi, max_tokens)
    state = make_state(ids, stoi, config)
    probes = load_probes(probes_path)
    curve = []
    done = 0
    t0 = time.time()
    for cp in sorted(set(checkpoints) | {steps}):
        if cp > steps:
            continue
        if cp > done:
            train_steps(state, cp - done)
            done = cp
        per = score_probes(state, probes)
        recent = state["losses"][-200:]
        curve.append({"step": done, **summarize(per),
                      "train_loss": sum(recent) / len(recent) if recent else None})
        print(f"[{name} s{seed}] step {done:>5}  probe acc {curve[-1]['choice_acc']:.1%}  "
              f"loss {curve[-1]['train_loss'] or float('nan'):.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    result = {"name": name, "seed": seed, "config": config, "steps": steps,
              "parameters": state["parameters"], "tokens": len(ids),
              "sentences_kept": len(kept), "unique_sentences": len(set(kept)),
              "sentences_dropped": len(dropped), "dropped_examples": dropped[:10],
              "curve": curve, "final_probes": per, "seconds": time.time() - t0}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_seed{seed}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
