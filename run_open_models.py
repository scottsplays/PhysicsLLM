#!/usr/bin/env python3
"""Run Qwen / Llama on this computer's NVIDIA GPU, same tests as run_eval.py.

  .venv\\Scripts\\python.exe run_open_models.py --models qwen2.5-1.5b
  .venv\\Scripts\\python.exe run_open_models.py --models qwen2.5-0.5b --limit 3   # 2-minute test
Options:
  --models   qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b, llama3.2-1b, llama3.2-3b (comma-separated)
  --sets     mcq,queries,safety (default: all three)
  --rag      on|off|both (default both)
  --batch    answers generated together (default 4; lowered automatically on out-of-memory)
  --dtype    float16 (default) or float32 (if answers come out as garbage / NaN)
  --limit    first N items of each set (0 = all)
  --redo-truncated --max-new 1500   re-run only the multiple-choice answers that
             ran out of room before the 'Answer: X' line

Answers go to evaluation/raw/<set>_<model>.jsonl as they finish; a rerun skips
what is done. Afterwards: run_eval.py --rebuild --set all
Llama needs HF_TOKEN in .env (Hugging Face token, license accepted).

One-time install of PyTorch with GPU support (pick cu124, or cu121/cu118 if
nvidia-smi reports an older "CUDA Version"):
  .venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
  .venv\\Scripts\\python.exe -m pip install transformers accelerate
"""

import argparse
import json
import sys
import time

import colab_eval as CE
import llm_clients


def drop_truncated(key, cap=500):
    """Move multiple-choice answers that ran out of tokens (no 'Answer: X')
    out of the log, into a .bak file, so the next run redoes them."""
    path = CE.EVAL / "raw" / f"mcq_{key}.jsonl"
    if not path.exists():
        return 0
    keep, drop = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        (drop if not r.get("choice") and r.get("output_tokens", 0) >= cap else keep).append(line)
    if drop:
        with open(path.with_name(path.name + ".truncated.bak"), "a", encoding="utf-8") as f:
            f.write("\n".join(drop) + "\n")
        path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return len(drop)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen2.5-1.5b")
    ap.add_argument("--sets", default="mcq,queries,safety")
    ap.add_argument("--rag", default="both", choices=["on", "off", "both"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true", help="run even without a GPU (very slow)")
    ap.add_argument("--max-new", type=int, default=0,
                    help="answer length limit in tokens for every set (default: mcq 512, queries 768, safety 512)")
    ap.add_argument("--redo-truncated", action="store_true",
                    help="re-run multiple-choice answers that hit the length limit before giving 'Answer: X'")
    a = ap.parse_args()
    if a.max_new:
        for k in CE.MAX_NEW:
            CE.MAX_NEW[k] = a.max_new

    try:
        import torch
    except ImportError:
        sys.exit("PyTorch is not installed in this .venv. See the install commands at the top of this file.")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
    elif not a.allow_cpu:
        sys.exit("No CUDA GPU visible to PyTorch. Either PyTorch was installed without GPU support "
                 "(reinstall with the CUDA command at the top of this file) or this computer has no NVIDIA GPU.")

    token = llm_clients._read_dotenv().get("HF_TOKEN") or None
    rags = {"on": (True,), "off": (False,), "both": (True, False)}[a.rag]
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    for key in [m.strip() for m in a.models.split(",") if m.strip()]:
        if key not in CE.OPEN_MODELS:
            sys.exit(f"Unknown model {key!r}. Choose from: {', '.join(CE.OPEN_MODELS)}")
        if key.startswith("llama") and not token:
            print(f"Skipping {key}: add HF_TOKEN=... to .env (license accepted on Hugging Face).")
            continue
        if a.redo_truncated:
            n = drop_truncated(key)
            print(f"{key}: {n} truncated multiple-choice answers set aside for a re-run")
            if n == 0:
                continue
        print(f"\n=== {key} ({CE.OPEN_MODELS[key]}): downloading/loading, first time takes a few minutes ===")
        t0 = time.time()
        model = CE.HFModel(CE.OPEN_MODELS[key], token=token, dtype=a.dtype)
        dev = next(model.model.parameters()).device
        print(f"Loaded on {dev}.")
        try:
            CE.run_model(key, model, sets=sets, rag_modes=rags, batch=a.batch, limit=a.limit)
        finally:
            model.close()
        print(f"{key} done in {(time.time() - t0) / 60:.0f} min")
    print("\nMultiple choice so far:")
    CE.quick_summary()
    print("\nNext: .venv\\Scripts\\python.exe run_eval.py --rebuild --set all")


if __name__ == "__main__":
    main()
