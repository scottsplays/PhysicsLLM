"""Train the shipped PhysicsSLM run and write physics_slm_run.json.

Recipe: 3 blocks, width 16, 4 heads, context 16, 1,000 steps, learning rate 0.05, seed 0.
Training windows stay in the first 90% of poc-corpus.txt. The file stores that run's
weights and the losses measured during the run.
"""

from pathlib import Path

import physics_slm_core as core

ROOT = Path(__file__).resolve().parent
CONFIG = {
    "block_size": 16,
    "n_embed": 16,
    "num_heads": 4,
    "n_blocks": 3,
    "seed": 0,
    "lr": 0.05,
}


def main():
    corpus = (ROOT / "poc-corpus.txt").read_text(encoding="utf-8").strip()
    state = core.initial_state(corpus, "poc-corpus.txt", CONFIG)
    state["target"] = 1000
    print(f"parameters {state['parameters']}", flush=True)
    print(f"tokens {state['tokens']} vocab {state['vocab_size']}", flush=True)
    print(f"train_last {core.train_last_start(state['tokens'], state['block_size'])}", flush=True)
    print(f"val_starts {core.held_out_starts(state['tokens'], state['block_size'])}", flush=True)

    def on_step(done, target, loss):
        if done == 1 or done % 50 == 0:
            print(f"step {done} loss {loss:.6f}", flush=True)

    core.train_steps(state, 1000, on_step)
    val = core.score_held_out(state)
    accuracy = state["hits"] / state["seen"]
    print(f"initial {state['losses'][0]:.6f}", flush=True)
    print(f"latest {state['losses'][-1]:.6f}", flush=True)
    print(f"val {val:.6f} windows {state['val_windows']}", flush=True)
    print(f"accuracy {accuracy:.6f}", flush=True)
    print(f"elapsed {state['elapsed']:.3f}", flush=True)
    print(f"params_after {core.count_parameters(state['model'])}", flush=True)
    for prompt in ("force equals", "gravity pulls"):
        result = core.generate(
            state["model"], state["stoi"], state["itos"], prompt, 3, state["block_size"], 0.0
        )
        print(f"GEN {result['full']}", flush=True)
    path = core.save_recorded_run(state, ROOT / core.RECORDED_RUN_NAME)
    loaded = core.load_recorded_run(path, corpus, "poc-corpus.txt")
    again = core.generate(
        loaded["model"], loaded["stoi"], loaded["itos"], "force equals", 3, loaded["block_size"], 0.0
    )
    print(f"RELOAD_GEN {again['full']}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
