"""Stage 0: does the hidden probe set tell a good curriculum from a bad one?

No LLM involved. Conditions, each trained twice (seeds 0 and 1):
  v2        your hand-written physicsslm_v2_train.txt
  shuffled  the same words in random order (same word counts, no sentences)
Step 0 of every run is the untrained model.

If v2 does not clearly beat shuffled on the probes, the probes cannot judge
teachers and must be fixed before Stage 1.

  python teach/stage0.py            (both conditions, 2 seeds, 3000 steps)
"""

import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TEACH = Path(__file__).resolve().parent
sys.path.insert(0, str(TEACH))

import teach_slm  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def curricula():
    v2 = (TEACH.parent / "physicsslm_v2_train.txt").read_text(encoding="utf-8")
    words = [w for w in v2.split() if w != "<eos>"]
    rng = random.Random(0)
    rng.shuffle(words)
    # Same words in random order, cut into "sentences" of the v2 average length.
    chunks, i = [], 0
    while i < len(words):
        n = rng.randint(4, 12)
        chunks.append(" ".join(words[i:i + n]))
        i += n
    return {"v2": v2, "shuffled": " <eos> ".join(chunks)}


def job(args):
    name, seed = args
    return teach_slm.run(name, curricula()[name], steps=STEPS, seed=seed)


if __name__ == "__main__":
    jobs = [(n, s) for s in (0, 1) for n in ("v2", "shuffled")]
    with ProcessPoolExecutor(2) as pool:
        results = list(pool.map(job, jobs))
    print("\nfinal probe accuracy (chance %.0f%%)" % (100 * results[0]["curve"][-1]["chance"]))
    for r in results:
        c = r["curve"][-1]
        print(f"  {r['name']:<9} seed {r['seed']}  choice {c['choice_acc']:.1%}  "
              f"top-1 {c['top1_acc']:.1%}  fact {c['acc_fact']:.0%}  "
              f"relation {c['acc_relation']:.0%}  novel {c['acc_novel']:.0%}")
