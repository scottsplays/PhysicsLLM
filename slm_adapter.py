"""PhysicsSLM (the from-scratch Transformer) as one of the compared models.

It does not edit or re-implement anything in mathematics.py, tensor.py,
Transformer.py, or dataloader.py. It loads the recorded run through
physics_slm_core and calls the model's own forward().

Two ways it takes part in the comparison:
  answer(question)          continuation of the in-vocabulary words of a
                            question (it cannot follow instructions)
  choose(question, options) multiple-choice by likelihood: the option
                            whose words the model finds most probable
                            after the question wins. This is how base
                            (non-instruction) language models are
                            usually scored on multiple-choice benchmarks.
"""

import math
import random
import re
from pathlib import Path

from physics_slm_core import RECORDED_RUN_NAME, generate, load_recorded_run

HERE = Path(__file__).resolve().parent
_STATE = None


def load():
    global _STATE
    if _STATE is None:
        corpus = (HERE / "poc-corpus.txt").read_text(encoding="utf-8").strip()
        _STATE = load_recorded_run(HERE / RECORDED_RUN_NAME, corpus, "poc-corpus.txt")
    return _STATE


def words_of(text):
    return re.findall(r"[a-z]+", (text or "").lower())


def in_vocab(text):
    stoi = load()["stoi"]
    return [w for w in words_of(text) if w in stoi]


def vocabulary():
    return sorted(load()["stoi"])


def answer(question, n_new=12, temperature=0.0):
    """Greedy continuation of the question's in-vocabulary words."""
    state = load()
    known = in_vocab(question)
    total = len(words_of(question))
    if not known:
        return {
            "text": ("None of these words are in my 32-word vocabulary, so I cannot "
                     "continue this prompt. I am a 7,648-parameter study model trained "
                     "for next-word prediction on template sentences; I cannot answer "
                     "questions."),
            "prompt_used": "", "coverage": 0.0, "known": [],
        }
    prompt = " ".join(known[-state["block_size"]:])
    result = generate(state["model"], state["stoi"], state["itos"], prompt,
                      n_new, state["block_size"], temperature)
    return {
        "text": result["full"],
        "prompt_used": prompt,
        "coverage": len(known) / max(1, total),
        "known": known,
        "top": result["top"],
    }


def _logprob_sequence(context_ids, target_ids):
    """Sum of log P(target_i | context + targets before i), via forward()."""
    state = load()
    model, block = state["model"], state["block_size"]
    ids = list(context_ids)
    total = 0.0
    for tok in target_ids:
        window = ids[-block:]
        probs = model.forward(window)
        row = probs[len(window) - 1].to_list()
        total += math.log(max(float(row[tok]), 1e-12))
        ids.append(tok)
    return total


def choose(question, options, seed=0):
    """Pick an option letter by mean per-word log-likelihood.

    Every word of an option counts. A word outside the vocabulary cannot be
    scored, so it gets the uniform log(1/V), the same as a random guess;
    an option is not rewarded for being mostly unknown words. Ties are
    broken at random (seeded), not by position, so the model is not
    credited for always picking 'A'.
    """
    state = load()
    stoi, V = state["stoi"], state["vocab_size"]
    uniform = math.log(1.0 / V)
    context = [stoi[w] for w in in_vocab(question)]
    scores, coverage = {}, {}
    for letter, text in options.items():
        opt = [stoi[w] for w in in_vocab(text)]
        n_words = max(1, len(words_of(text)))
        n_unknown = n_words - len(opt)
        coverage[letter] = len(opt) / n_words
        if not opt:
            scores[letter] = uniform
            continue
        if context:
            known_lp = _logprob_sequence(context, opt)
        else:  # nothing to condition on: first word is a uniform guess
            known_lp = uniform + _logprob_sequence(opt[:1], opt[1:])
        scores[letter] = (known_lp + n_unknown * uniform) / n_words
    best = max(scores.values())
    tied = sorted(k for k, v in scores.items() if abs(v - best) < 1e-9)
    pick = random.Random(f"{seed}-{question}").choice(tied)
    return {"choice": pick, "scores": scores, "coverage": coverage,
            "tied": len(tied) > 1, "question_words_known": len(context)}
