"""Evaluate open-weight models (Qwen, Llama) on the same test sets as run_eval.py.

Runs in Google Colab on a GPU (see eval_open_models.ipynb). Retrieval, tutor
rules, the poisoned-excerpt test and multiple-choice parsing are the SAME code
the hosted models used (rag.py, tutor.py, llm_clients.parse_choice), so the
results are directly comparable.

Each answer is appended to <out_dir>/<set>_<model>.jsonl as soon as it is
generated, in run_eval.py's record format. A rerun skips what is already done.
Copy the .jsonl files into evaluation/raw/ and run
    python run_eval.py --rebuild
to merge them into the CSVs, the app's Results page and the summary.
"""

import csv
import gc
import json
import time
from pathlib import Path

import llm_clients
import rag
import tutor

HERE = Path(__file__).resolve().parent
EVAL = HERE / "evaluation"

OPEN_MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}
MAX_NEW = {"mcq": 512, "queries": 768, "safety": 512}
TEMPERATURE = 0.2  # same as the hosted runs


# ------------------------------------------------------------------ test items

def _csv(name):
    with open(EVAL / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def items_for(set_name):
    if set_name == "mcq":
        rows = _csv("mcq_workbook.csv")
        for r in rows:
            r["options"] = {k: r[k] for k in "ABCDE" if r.get(k)}
        return rows
    if set_name == "queries":
        return _csv("test_queries.csv")
    return _csv("safety_tests.csv")


def build_prompt(set_name, item, use_rag, k=4):
    """(system, user, source labels): mirrors tutor.ask() exactly."""
    options = item.get("options") if set_name == "mcq" else None
    question = item["question"] if set_name == "mcq" else (item.get("query") or item["prompt"])
    hits = None
    if use_rag:
        hits = rag.retrieve(question if not options else question + " " + " ".join(options.values()), k)
        if set_name == "safety" and item.get("poisoned_excerpt"):
            fake = rag.Chunk(-1, "Poisoned test excerpt (in memory only)", "test-only", item["poisoned_excerpt"])
            hits = hits[:-1] + [(fake, 0.0)] if hits else [(fake, 0.0)]
    system = tutor.system_prompt(use_rag) + (tutor.MCQ_SUFFIX if options else "")
    user = tutor.build_user_message(question, hits, None, options)
    return system, user, [c.label for c, _ in (hits or [])]


# ------------------------------------------------------------------ model

class HFModel:
    """A chat model from the Hugging Face hub; batched generation sampled at T=0.2."""

    def __init__(self, repo, token=None, dtype="float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, token=token)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, token=token, torch_dtype=getattr(torch, dtype), device_map="auto")
        self.model.eval()

    def _text(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, pairs, max_new_tokens, seed=0):
        """pairs: [(system, user)]. Returns [(text, n_in, n_out)]."""
        torch = self.torch
        texts = [self._text(s, u) for s, u in pairs]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=TEMPERATURE, top_p=1.0, pad_token_id=self.tok.pad_token_id)
        width = enc["input_ids"].shape[1]
        results = []
        for i in range(len(texts)):
            new = out[i, width:]
            n_out = int((new != self.tok.pad_token_id).sum())
            n_in = int(enc["attention_mask"][i].sum())
            results.append((self.tok.decode(new, skip_special_tokens=True).strip(), n_in, n_out))
        return results

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


# ------------------------------------------------------------------ runner

def _done(path):
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    keys.add((r["id"], r["rag"]))
    return keys


def run_model(model_key, gen, sets=("mcq", "queries", "safety"), rag_modes=(True, False),
              out_dir=EVAL / "raw", batch=8, limit=0, log=print):
    """Run every (item, rag) pair not already saved. gen: an HFModel (or any
    object with .generate(pairs, max_new_tokens) -> [(text, n_in, n_out)])."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for set_name in sets:
        path = out_dir / f"{set_name}_{model_key}.jsonl"
        done = _done(path)
        jobs = []
        for it in items_for(set_name)[: limit or None]:
            for use_rag in rag_modes:
                if (it["id"], use_rag) not in done:
                    jobs.append((it, use_rag))
        log(f"[{model_key}] {set_name}: {len(jobs)} to run ({len(done)} already saved)")
        b = batch
        i = 0
        while i < len(jobs):
            chunk = jobs[i:i + b]
            prompts = [build_prompt(set_name, it, r) for it, r in chunk]
            t0 = time.time()
            try:
                outs = gen.generate([(s, u) for s, u, _ in prompts], MAX_NEW[set_name])
            except RuntimeError as e:  # CUDA out of memory: halve the batch and retry
                if "out of memory" in str(e).lower() and b > 1:
                    b = max(1, b // 2)
                    gen.torch.cuda.empty_cache()
                    log(f"  out of memory, batch size -> {b}")
                    continue
                raise
            per = (time.time() - t0) / len(chunk)
            with open(path, "a", encoding="utf-8") as f:
                for (it, use_rag), (_, _, labels), (text, n_in, n_out) in zip(chunk, prompts, outs):
                    rec = {"id": it["id"], "model": model_key, "rag": use_rag, "answer": text,
                           "choice": llm_clients.parse_choice(text) if set_name == "mcq" else "",
                           "sources": labels, "seconds": round(per, 2), "input_tokens": n_in,
                           "output_tokens": n_out, "error": "",
                           "notes": f"local {OPEN_MODELS.get(model_key, model_key)}, T={TEMPERATURE}",
                           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            i += len(chunk)
            log(f"  {set_name} {i}/{len(jobs)}  ({per:.1f}s per answer)")


def quick_summary(out_dir=EVAL / "raw"):
    """Multiple-choice accuracy per model and RAG setting, from the saved files."""
    answer = {r["id"]: r["answer"] for r in items_for("mcq")}
    rows = []
    for path in sorted(Path(out_dir).glob("mcq_*.jsonl")):
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for use_rag in (True, False):
            rs = [r for r in recs if r["rag"] == use_rag and not r["error"]]
            if rs:
                ok = sum(r["choice"] == answer[r["id"]] for r in rs)
                unparsed = sum(1 for r in rs if not r["choice"])
                rows.append((rs[0]["model"], "on" if use_rag else "off", ok, len(rs), unparsed))
    for m, rag_on, ok, n, un in rows:
        print(f"{m:<14} RAG {rag_on:<3}  {ok}/{n} = {ok / n:.0%}   (no 'Answer:' line: {un})")
    return rows
