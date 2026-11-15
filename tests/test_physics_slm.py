"""Checks for the PhysicsSLM app layer. Does not import or run dataloader.py."""

import math
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import mathematics
from physics_slm_core import (
    causal_report,
    compatible_prompts,
    corpus_split,
    count_parameters,
    evaluate_validation,
    generate,
    generation_matches,
    held_out_starts,
    initial_state,
    load_recorded_run,
    moving_average,
    parameter_count,
    remember_validation,
    save_recorded_run,
    score_held_out,
    to_rows,
    trace_forward,
    train_last_start,
    train_steps,
    uniform_loss,
)
from physics_slm_playground import (
    LIBRARY_FILES,
    component_names,
    extract_symbol,
    list_symbols,
    read_library,
    run_component,
    spec_for,
)
from tensor import Matrix
from Transformer import SingleHeadAttention, Transformer

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    "Embedding",
    "PositionalEncoding",
    "Linear",
    "ReLU",
    "FeedForward",
    "Softmax",
    "LayerNorm",
    "SingleHeadAttention",
    "MultiHeadAttention",
    "TransformerBlock",
    "Transformer",
    "CrossEntropyLoss",
]


def toy_config(**overrides):
    config = {
        "block_size": 4,
        "n_embed": 8,
        "num_heads": 2,
        "n_blocks": 1,
        "seed": 0,
        "lr": 0.5,
    }
    config.update(overrides)
    return config


class PhysicsSLMTests(unittest.TestCase):
    def test_library_files_and_symbols(self):
        self.assertNotIn("layer.py", LIBRARY_FILES)
        source = read_library("Transformer.py")
        names = {name for _kind, name in list_symbols(source)}
        for name in CLASSES:
            self.assertIn(name, names)
            body, start, end = extract_symbol(source, name)
            self.assertGreaterEqual(start, 1)
            self.assertGreater(end, start)
            self.assertTrue(body.startswith(f"class {name}"))
        for path in ("physics_slm_core.py", "physics_slm_playground.py", "physics_slm_app.py", "prompt_service.py"):
            text = (ROOT / path).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"(?m)^\s*(?:import|from)\s+dataloader\b")

    def test_every_component_runs(self):
        self.assertEqual(len(CLASSES), 12)
        for name in CLASSES:
            self.assertIsNotNone(spec_for(name))
        for name in component_names():
            result = run_component(name)
            self.assertIn(result["kind"], {"random", "fixed"})
            self.assertTrue(str(result["text"]))
            if spec_for(name)["backward"]:
                backward = run_component(name, backward=True)
                self.assertIn(backward["kind"], {"random", "fixed"})
                self.assertNotEqual(backward["kind"], "trained")

    def test_invalid_input_is_an_error(self):
        with self.assertRaises(ValueError):
            run_component("relu", {"text": "nope"})
        with self.assertRaises(ValueError):
            run_component("Embedding", {"text": "-1, 2"})
        with self.assertRaises(ValueError):
            run_component("Linear", {"text": "1, 2\n3"})

    def test_vector_add_and_causal_tril(self):
        added = run_component("Vector")
        self.assertIn("5.0000", added["text"])
        masked = run_component("Matrix", {"op": "tril", "left": "1, 2\n3, 4"})
        self.assertIn("-inf", masked["text"])

    def test_attention_mask_is_the_models_softmax(self):
        np.random.seed(0)
        head = SingleHeadAttention(8, 4)
        head.forward(Matrix([[0.1] * 8 for _ in range(5)]))
        probs = to_rows(head.sm.predicted)
        report = causal_report(probs)
        self.assertEqual(len(probs), 5)
        self.assertEqual(len(probs[0]), 5)
        self.assertLess(report["max_future"], 1e-8)
        for total in report["row_sums"]:
            self.assertAlmostEqual(total, 1.0, places=5)
        demo = run_component("SingleHeadAttention")
        future = float(dict(demo["details"])["max future probability"])
        self.assertLess(future, 1e-6)

    def test_trace_matches_forward_and_is_causal(self):
        np.random.seed(1)
        model = Transformer(6, 8, 2, 4, 1)
        ids = [1, 0, 2, 1]
        trace = trace_forward(model, ids, ["a", "b", "c", "d"])
        again = to_rows(model.forward(ids))
        for left, right in zip(trace["probs"], again):
            for actual, expected in zip(left, right):
                self.assertAlmostEqual(actual, expected, places=5)
        for block in trace["blocks"]:
            self.assertEqual(len(block["heads"]), 2)
            for head in block["heads"]:
                self.assertEqual(len(head["Q"][0]), head["head_size"])
                self.assertLess(causal_report(head["probs"])["max_future"], 1e-8)
                self.assertTrue(any(math.isinf(value) and value < 0 for row in head["scores"] for value in row))

    def test_training_generation_and_validation(self):
        corpus = "alpha beta gamma delta epsilon " * 6
        state = initial_state(corpus.strip(), "toy", toy_config())
        self.assertEqual(
            state["parameters"],
            parameter_count(state["vocab_size"], 8, 2, 4, 1),
        )
        self.assertEqual(state["parameters"], count_parameters(state["model"]))
        train_steps(state, 2)
        self.assertEqual(len(state["losses"]), 2)
        self.assertEqual(state["steps_done"], 2)
        self.assertTrue(all(math.isfinite(loss) for loss in state["losses"]))
        self.assertGreater(state["seen"], 0)
        produced = generate(state["model"], state["stoi"], state["itos"], "alpha beta", 2, state["block_size"], 0)
        self.assertEqual(produced["prompt"], "alpha beta")
        self.assertEqual(len(produced["continuation"].split()), 2)
        self.assertTrue(produced["full"].startswith("alpha beta "))
        self.assertEqual(len(produced["top"]), 5)
        with self.assertRaises(ValueError):
            generate(state["model"], state["stoi"], state["itos"], "notaword", 1, state["block_size"], 0)
        self.assertEqual(compatible_prompts({"alpha": 0, "beta": 1}), [])
        self.assertIn("force equals", compatible_prompts({"force": 0, "equals": 1, "gravity": 2, "pulls": 3}))
        remember_validation(state, "alpha beta gamma delta epsilon zeta " * 3)
        self.assertGreater(state["val_dropped"], 0)
        loss = evaluate_validation(state, windows=2)
        self.assertTrue(math.isfinite(loss))
        self.assertIsNotNone(state["file_val_accuracy"])
        self.assertIsNone(state["val_loss"])
        self.assertEqual(len(moving_average(state["losses"], 20)), 2)
        self.assertAlmostEqual(uniform_loss(state["vocab_size"]), math.log(state["vocab_size"]))
        produced_again = generate(state["model"], state["stoi"], state["itos"], "alpha beta", 2, state["block_size"], 0.1)
        self.assertTrue(generation_matches(produced_again, "alpha beta", 0.1, 2))
        self.assertFalse(generation_matches(produced_again, "alpha beta", 0.0, 2))
        self.assertFalse(generation_matches(produced_again, "gamma delta", 0.1, 2))

    def test_three_block_parameter_count(self):
        np.random.seed(0)
        model = Transformer(32, 16, 4, 16, 3)
        self.assertEqual(model.pos_encoding.pos_emb.embedding_table.shape, (16, 16))
        self.assertEqual(count_parameters(model), 7648)
        self.assertEqual(parameter_count(32, 16, 4, 16, 3), 7648)

    def test_train_step_does_not_chain_output_softmax_backward(self):
        corpus = "alpha beta gamma delta epsilon " * 6
        state = initial_state(corpus.strip(), "toy", toy_config())
        calls = []
        original = state["model"].sm.backward

        def wrapped(output):
            calls.append(True)
            return original(output)

        state["model"].sm.backward = wrapped
        train_steps(state, 1)
        self.assertEqual(calls, [])
        probabilities = state["model"].forward([0, 1, 2, 3])
        for row in probabilities:
            self.assertAlmostEqual(sum(float(value) for value in row.to_list()), 1.0, places=5)

    def test_held_out_tail_is_forward_only(self):
        corpus = " ".join(f"w{index % 11}" for index in range(200))
        state = initial_state(corpus, "toy", toy_config())
        seen = []
        real = mathematics.random_int

        def capture(low, high):
            seen.append((low, high))
            return high

        mathematics.random_int = capture
        try:
            train_steps(state, 1)
        finally:
            mathematics.random_int = real
        split = corpus_split(len(state["ids"]))
        self.assertEqual(seen, [(0, train_last_start(len(state["ids"]), state["block_size"]))])
        self.assertLess(seen[0][1] + state["block_size"], split)
        before = to_rows(state["model"].embedding.embedding_table)
        loss = score_held_out(state)
        self.assertEqual(state["val_windows"], 8)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(all(start >= split for start in state["val_starts"]))
        self.assertEqual(before, to_rows(state["model"].embedding.embedding_table))
        starts = held_out_starts(1425, 16, 8)
        self.assertEqual(len(starts), 8)
        self.assertGreaterEqual(min(starts), corpus_split(1425))

    def test_recorded_run_roundtrip(self):
        corpus = " ".join(f"w{index % 11}" for index in range(200))
        state = initial_state(corpus, "toy", toy_config())
        train_steps(state, 1)
        score_held_out(state)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            save_recorded_run(state, path)
            loaded = load_recorded_run(path, corpus, "toy")
        self.assertEqual(loaded["losses"], state["losses"])
        self.assertEqual(loaded["parameters"], count_parameters(loaded["model"]))
        self.assertAlmostEqual(loaded["val_loss"], state["val_loss"])
        left = to_rows(state["model"].forward([0, 1, 2, 3]))
        right = to_rows(loaded["model"].forward([0, 1, 2, 3]))
        for left_row, right_row in zip(left, right):
            for actual, expected in zip(left_row, right_row):
                self.assertAlmostEqual(actual, expected, places=6)

    def test_checkpoint_roundtrip(self):
        corpus = "alpha beta gamma delta epsilon " * 4
        state = initial_state(corpus.strip(), "toy", toy_config())
        train_steps(state, 1)
        state["running"] = False
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pkl"
            with path.open("wb") as handle:
                pickle.dump({"version": 1, "state": state}, handle)
            with path.open("rb") as handle:
                loaded = pickle.load(handle)["state"]
        self.assertEqual(loaded["steps_done"], 1)
        self.assertEqual(count_parameters(loaded["model"]), state["parameters"])
        self.assertEqual(loaded["losses"], state["losses"])


if __name__ == "__main__":
    unittest.main()
