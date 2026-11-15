"""Training, generation, and activation tracing for the handwritten Transformer.

This module does not modify mathematics.py, tensor.py, or Transformer.py.
It never imports dataloader.py, which trains a model as soon as it is imported.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import mathematics
from tensor import Matrix
from Transformer import CrossEntropyLoss, Transformer

EXAMPLES = (
    "force equals",
    "gravity pulls",
    "kinetic energy",
    "momentum equals",
    "the ball",
)
CHECKPOINT_NAME = "physics_slm.pkl"
RECORDED_RUN_NAME = "physics_slm_run.json"


def to_rows(matrix):
    """Copy a Matrix into plain floats so later passes cannot change the snapshot."""
    return [[float(value) for value in row.to_list()] for row in matrix]


def shape_of(rows):
    if not rows:
        return (0, 0)
    return (len(rows), len(rows[0]))


def one_hot(targets, vocab_size):
    rows = []
    for token in targets:
        row = [0.0] * vocab_size
        row[int(token)] = 1.0
        rows.append(row)
    return rows


def moving_average(values, window=20):
    if window < 1:
        raise ValueError("The average window has to be at least 1.")
    totals = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
            totals.append(running / window)
        else:
            totals.append(running / (index + 1))
    return totals


def uniform_loss(vocab_size):
    """Mean cross-entropy of a one-hot target under a uniform distribution."""
    if vocab_size < 2:
        raise ValueError("Vocabulary size has to be at least 2.")
    return math.log(vocab_size)


def parameter_count(vocab_size, n_embed, num_heads, block_size, n_blocks):
    """Count weights from the layer shapes, including both embedding tables.

    Linear layers have no bias. Each layer norm contributes a gamma and a beta.
    The count matches the live Matrix and Vector objects in count_parameters.
    """
    if n_embed % num_heads != 0:
        raise ValueError("Embedding width has to be divisible by the number of heads.")
    if min(vocab_size, n_embed, num_heads, block_size, n_blocks) < 1:
        raise ValueError("Model sizes have to be positive.")
    width = n_embed
    per_block = 4 * width + 8 * width * width
    return 2 * vocab_size * width + block_size * width + 2 * width + n_blocks * per_block


def count_parameters(model):
    total = 0
    tables = [
        model.embedding.embedding_table,
        model.pos_encoding.pos_emb.embedding_table,
        model.unembedding.weights,
        model.ln.gamma,
        model.ln.beta,
    ]
    for block in model.blocks:
        tables.extend([block.ln1.gamma, block.ln1.beta, block.ln2.gamma, block.ln2.beta])
        tables.append(block.attention.proj.weights)
        tables.extend([block.ff.l1.weights, block.ff.l2.weights])
        for head in block.attention.heads:
            tables.extend([head.w_queries.weights, head.w_keys.weights, head.w_values.weights])
    for table in tables:
        if hasattr(table, "m"):
            total += table.n * table.m
        else:
            total += table.n
    return total


def fingerprint(corpus, source, config):
    digest = hashlib.sha1(corpus.encode("utf-8")).hexdigest()
    return (
        digest,
        source,
        int(config["block_size"]),
        int(config["n_embed"]),
        int(config["num_heads"]),
        int(config["n_blocks"]),
        int(config["seed"]),
    )


def initial_state(corpus, source, config):
    words = corpus.split()
    block_size = int(config["block_size"])
    if len(words) <= block_size:
        raise ValueError(
            f"The corpus needs more than {block_size} words so every window has a next token."
        )
    vocab = sorted(set(words))
    stoi = {word: index for index, word in enumerate(vocab)}
    itos = {index: word for word, index in stoi.items()}
    ids = [stoi[word] for word in words]
    n_embed = int(config["n_embed"])
    num_heads = int(config["num_heads"])
    n_blocks = int(config["n_blocks"])
    if n_embed % num_heads != 0:
        raise ValueError("Embedding width has to be divisible by the number of heads.")
    np.random.seed(int(config["seed"]))
    model = Transformer(len(vocab), n_embed, num_heads, block_size, n_blocks)
    return {
        "model": model,
        "stoi": stoi,
        "itos": itos,
        "ids": ids,
        "vocab_size": len(vocab),
        "tokens": len(ids),
        "losses": [],
        "hits": 0,
        "seen": 0,
        "steps_done": 0,
        "target": 0,
        "lr": float(config["lr"]),
        "seed": int(config["seed"]),
        "block_size": block_size,
        "n_embed": n_embed,
        "num_heads": num_heads,
        "n_blocks": n_blocks,
        "source": source,
        "fingerprint": fingerprint(corpus, source, config),
        "elapsed": 0.0,
        "running": False,
        "parameters": count_parameters(model),
        "rng": np.random.get_state(),
        "val_ids": None,
        "val_dropped": 0,
        "val_loss": None,
        "val_accuracy": None,
        "val_windows": 0,
        "val_scored_at": None,
    }


def corpus_split(n_tokens):
    """Index where the held-out tail starts. Training never reads this index or anything after it."""
    return int(n_tokens * 0.9)


def train_last_start(n_tokens, block):
    """Last legal training-window start. The window's target token stays before the tail."""
    last = corpus_split(n_tokens) - int(block) - 1
    if last < 0:
        raise ValueError(
            "The corpus is too short to keep the last 10% of tokens out of training."
        )
    return last


def held_out_starts(n_tokens, block, windows=8):
    """Fixed starts in the last 10% of tokens. Empty when a window does not fit in that tail."""
    first = corpus_split(n_tokens)
    last = int(n_tokens) - int(block) - 1
    if last < first:
        return []
    count = min(int(windows), last - first + 1)
    if count <= 1:
        return [first]
    span = last - first
    starts = [first + (i * span) // (count - 1) for i in range(count)]
    unique = []
    for start in starts:
        if not unique or start != unique[-1]:
            unique.append(start)
    return unique


def train_steps(state, n, on_step=None):
    """Train on random next-token windows from the first 90% of tokens.

    `Transformer.forward` already applies the output softmax, so `probabilities`
    is a probability matrix. `CrossEntropyLoss.forward` scores those probabilities.
    `CrossEntropyLoss.backward` returns (p - y) / T, the combined softmax-plus-loss
    gradient, and that matrix is passed straight to `Transformer.backward`.
    The output softmax's own backward is not called after the loss.
    """
    n = int(n)
    if n < 1:
        return
    np.random.set_state(state["rng"])
    model = state["model"]
    ids = state["ids"]
    block = state["block_size"]
    vocab = state["vocab_size"]
    loss_fn = CrossEntropyLoss()
    lr = float(state["lr"])
    last_start = train_last_start(len(ids), block)
    started = time.perf_counter()
    for _ in range(n):
        start = mathematics.random_int(0, last_start)
        context = ids[start : start + block]
        targets = ids[start + 1 : start + block + 1]
        probabilities = model.forward(context)
        loss = float(loss_fn.forward(probabilities, one_hot(targets, vocab)))
        hits = 0
        for row, target in zip(probabilities, targets):
            if row.argmax() == target:
                hits += 1
        model.backward(loss_fn.backward(), lr)
        state["losses"].append(loss)
        state["hits"] += hits
        state["seen"] += block
        state["steps_done"] += 1
        if on_step is not None:
            on_step(state["steps_done"], state["target"], loss)
    state["elapsed"] += time.perf_counter() - started
    state["rng"] = np.random.get_state()


def score_held_out(state, windows=8):
    """Forward-only mean loss on fixed windows from the last 10% of the training corpus.

    Does not sample, does not call backward, and does not change the weights.
    """
    ids = state["ids"]
    block = state["block_size"]
    starts = held_out_starts(len(ids), block, windows)
    state["val_starts"] = list(starts)
    state["val_windows"] = len(starts)
    state["val_scored_at"] = state["steps_done"]
    if not starts:
        state["val_loss"] = None
        return None
    loss_fn = CrossEntropyLoss()
    losses = []
    for start in starts:
        context = ids[start : start + block]
        targets = ids[start + 1 : start + block + 1]
        probabilities = state["model"].forward(context)
        losses.append(float(loss_fn.forward(probabilities, one_hot(targets, state["vocab_size"]))))
    state["val_loss"] = sum(losses) / len(losses)
    return state["val_loss"]


def remember_validation(state, text):
    """Store an optional extra file. This does not replace the held-out tail score."""
    words = text.split()
    kept = [state["stoi"][word] for word in words if word in state["stoi"]]
    state["val_ids"] = kept
    state["val_dropped"] = len(words) - len(kept)
    state["file_val_loss"] = None
    state["file_val_accuracy"] = None
    state["file_val_windows"] = 0
    return len(kept)


def evaluate_validation(state, windows=8):
    """Score an uploaded file with forward passes only. Does not update weights or the tail loss."""
    ids = state.get("val_ids") or []
    block = state["block_size"]
    if len(ids) <= block:
        raise ValueError(
            "The validation file needs more in-vocabulary words than the context length."
        )
    rng = np.random.RandomState(state["seed"])
    span = len(ids) - block - 1
    count = min(int(windows), span + 1)
    starts = sorted({int(value) for value in rng.randint(0, span + 1, size=count)})
    loss_fn = CrossEntropyLoss()
    losses = []
    hits = 0
    seen = 0
    for start in starts:
        context = ids[start : start + block]
        targets = ids[start + 1 : start + block + 1]
        probabilities = state["model"].forward(context)
        losses.append(float(loss_fn.forward(probabilities, one_hot(targets, state["vocab_size"]))))
        for row, target in zip(probabilities, targets):
            if row.argmax() == target:
                hits += 1
            seen += 1
    state["file_val_loss"] = sum(losses) / len(losses)
    state["file_val_accuracy"] = hits / seen if seen else None
    state["file_val_windows"] = len(starts)
    return state["file_val_loss"]


def window_accuracy(state):
    if not state["seen"]:
        return None
    return state["hits"] / state["seen"]


def compatible_prompts(stoi, prompts=EXAMPLES):
    ready = []
    for prompt in prompts:
        words = prompt.split()
        if words and all(word in stoi for word in words):
            ready.append(prompt)
    return ready


def choose_token(probabilities, temperature, rng):
    probs = np.asarray(list(probabilities), dtype=float)
    if temperature <= 0 or len(probs) == 1:
        return int(np.argmax(probs))
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logp / float(temperature)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return int(np.argmax(probs))
    weights /= total
    return int(rng.choice(len(weights), p=weights))


def top_tokens(row, itos, k=5):
    ranked = sorted(((float(prob), index) for index, prob in enumerate(row)), reverse=True)
    return [(itos[index], prob) for prob, index in ranked[:k]]


def generate(model, stoi, itos, prompt, n_new, block_size, temperature=0.0):
    """Autoregressive continuation. Each new token is chosen from the last row of one forward."""
    words = prompt.split()
    if not words:
        raise ValueError("Enter a prompt.")
    missing = sorted({word for word in words if word not in stoi})
    if missing:
        raise ValueError("These words are not in the vocabulary: " + ", ".join(missing))
    n_new = int(n_new)
    if n_new < 1:
        raise ValueError("Generate at least one token.")
    ids = [stoi[word] for word in words]
    note = None
    if len(ids) > block_size:
        note = (
            f"The prompt is longer than the context window of {block_size}. "
            f"Only the last {block_size} words are read."
        )
    rng = np.random.RandomState()
    first_row = None
    chosen = []
    for step in range(n_new):
        window = ids[-block_size:]
        probabilities = model.forward(window)
        row = [float(value) for value in probabilities[len(window) - 1].to_list()]
        if step == 0:
            first_row = row
        ids.append(choose_token(row, temperature, rng))
        chosen.append(ids[-1])
    prompt_text = " ".join(words)
    continuation = " ".join(itos[token] for token in chosen)
    return {
        "prompt": prompt_text,
        "continuation": continuation,
        "full": f"{prompt_text} {continuation}".strip(),
        "top": top_tokens(first_row, itos, 5),
        "note": note,
        "temperature": float(temperature),
        "n_new": n_new,
    }


def generation_matches(result, prompt, temperature, n_new):
    """True only when the stored continuation was produced for these controls."""
    if not result:
        return False
    words = str(prompt).split()
    if result.get("prompt") != " ".join(words):
        return False
    if int(result.get("n_new", -1)) != int(n_new):
        return False
    return abs(float(result.get("temperature", -999.0)) - float(temperature)) <= 1e-9


def causal_report(probs):
    """Future positions are columns j > i. Their attention should be numerically zero."""
    largest = 0.0
    sums = []
    for i, row in enumerate(probs):
        sums.append(float(sum(row)))
        for j, value in enumerate(row):
            if j > i and math.isfinite(value):
                largest = max(largest, abs(float(value)))
    return {"max_future": largest, "row_sums": sums}


def trace_forward(model, token_ids, tokens=None):
    """Same calls as Transformer.forward, with a copy of every activation.

    SingleHeadAttention already keeps Q, K, V, the masked scores, and the
    softmax probabilities. Those copies are what the heatmaps draw.
    """
    ids = [int(token) for token in token_ids]
    if not ids:
        raise ValueError("Enter at least one token.")
    if len(ids) > model.block_size:
        raise ValueError(
            f"The context window is {model.block_size} tokens. This sequence has {len(ids)}."
        )
    token_emb = model.embedding.forward(ids)
    pos_emb = model.pos_encoding.forward(ids)
    embedded = token_emb + pos_emb
    token_rows = to_rows(token_emb)
    pos_rows = to_rows(pos_emb)
    embedded_rows = to_rows(embedded)
    x = embedded
    blocks = []
    for block in model.blocks:
        ln1 = block.ln1.forward(x)
        attn = block.attention.forward(ln1)
        residual1 = x + attn
        ln2 = block.ln2.forward(residual1)
        ff = block.ff.forward(ln2)
        out = residual1 + ff
        heads = []
        for head in block.attention.heads:
            heads.append(
                {
                    "Q": to_rows(head.queries),
                    "K": to_rows(head.keys),
                    "V": to_rows(head.values),
                    "scores": to_rows(head.wei),
                    "probs": to_rows(head.sm.predicted),
                    "output": to_rows(head.output),
                    "scale": float(head.scale),
                    "head_size": int(head.w_values.output_dim),
                }
            )
        blocks.append(
            {
                "ln1": to_rows(ln1),
                "attention": to_rows(attn),
                "residual1": to_rows(residual1),
                "ln2": to_rows(ln2),
                "feedforward": to_rows(ff),
                "relu": to_rows(block.ff.ReLU.output),
                "output": to_rows(out),
                "heads": heads,
            }
        )
        x = out
    final_ln = model.ln.forward(x)
    logits = model.unembedding.forward(final_ln)
    probs = model.sm.forward(logits)
    return {
        "ids": ids,
        "tokens": list(tokens) if tokens is not None else [str(token) for token in ids],
        "token_embeddings": token_rows,
        "position_embeddings": pos_rows,
        "embedded": embedded_rows,
        "blocks": blocks,
        "final_ln": to_rows(final_ln),
        "logits": to_rows(logits),
        "probs": to_rows(probs),
        "n_embed": int(model.embedding.embedding_table.m),
        "block_size": int(model.block_size),
        "vocab_size": int(model.unembedding.output_dim),
        "embedding_table": (
            int(model.embedding.embedding_table.n),
            int(model.embedding.embedding_table.m),
        ),
    }


def _write_matrix(matrix, rows):
    if len(rows) != matrix.n or (rows and len(rows[0]) != matrix.m):
        raise ValueError("A weight table in the recorded run does not match this model.")
    for index, row in enumerate(rows):
        matrix.vectors[index].vals = [float(value) for value in row]
        matrix.vectors[index].n = len(row)


def _write_vector(vector, values):
    if len(values) != vector.n:
        raise ValueError("A vector in the recorded run does not match this model.")
    vector.vals = [float(value) for value in values]
    vector.n = len(values)


def _vector_list(vector):
    return [float(value) for value in vector.to_list()]


def export_weights(model):
    blocks = []
    for block in model.blocks:
        blocks.append(
            {
                "heads": [
                    {
                        "q": to_rows(head.w_queries.weights),
                        "k": to_rows(head.w_keys.weights),
                        "v": to_rows(head.w_values.weights),
                    }
                    for head in block.attention.heads
                ],
                "proj": to_rows(block.attention.proj.weights),
                "l1": to_rows(block.ff.l1.weights),
                "l2": to_rows(block.ff.l2.weights),
                "ln1_gamma": _vector_list(block.ln1.gamma),
                "ln1_beta": _vector_list(block.ln1.beta),
                "ln2_gamma": _vector_list(block.ln2.gamma),
                "ln2_beta": _vector_list(block.ln2.beta),
            }
        )
    return {
        "embedding": to_rows(model.embedding.embedding_table),
        "position": to_rows(model.pos_encoding.pos_emb.embedding_table),
        "unembedding": to_rows(model.unembedding.weights),
        "ln_gamma": _vector_list(model.ln.gamma),
        "ln_beta": _vector_list(model.ln.beta),
        "blocks": blocks,
    }


def apply_weights(model, weights):
    _write_matrix(model.embedding.embedding_table, weights["embedding"])
    _write_matrix(model.pos_encoding.pos_emb.embedding_table, weights["position"])
    _write_matrix(model.unembedding.weights, weights["unembedding"])
    _write_vector(model.ln.gamma, weights["ln_gamma"])
    _write_vector(model.ln.beta, weights["ln_beta"])
    if len(weights["blocks"]) != len(model.blocks):
        raise ValueError("The recorded run has a different number of blocks.")
    for block, saved in zip(model.blocks, weights["blocks"]):
        if len(saved["heads"]) != len(block.attention.heads):
            raise ValueError("The recorded run has a different number of heads.")
        for head, saved_head in zip(block.attention.heads, saved["heads"]):
            _write_matrix(head.w_queries.weights, saved_head["q"])
            _write_matrix(head.w_keys.weights, saved_head["k"])
            _write_matrix(head.w_values.weights, saved_head["v"])
        _write_matrix(block.attention.proj.weights, saved["proj"])
        _write_matrix(block.ff.l1.weights, saved["l1"])
        _write_matrix(block.ff.l2.weights, saved["l2"])
        _write_vector(block.ln1.gamma, saved["ln1_gamma"])
        _write_vector(block.ln1.beta, saved["ln1_beta"])
        _write_vector(block.ln2.gamma, saved["ln2_gamma"])
        _write_vector(block.ln2.beta, saved["ln2_beta"])


def _dump_rng(rng_state):
    kind, keys, pos, has_gauss, cached = rng_state
    return {
        "kind": str(kind),
        "keys": [int(value) for value in keys],
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached": float(cached),
    }


def _load_rng(payload):
    keys = np.array(payload["keys"], dtype=np.uint32)
    return (str(payload["kind"]), keys, int(payload["pos"]), int(payload["has_gauss"]), float(payload["cached"]))


def recorded_payload(state):
    return {
        "version": 1,
        "fingerprint": list(state["fingerprint"]),
        "stoi": state["stoi"],
        "config": {
            "block_size": state["block_size"],
            "n_embed": state["n_embed"],
            "num_heads": state["num_heads"],
            "n_blocks": state["n_blocks"],
            "seed": state["seed"],
            "lr": state["lr"],
        },
        "source": state["source"],
        "steps_done": state["steps_done"],
        "target": state["target"],
        "losses": state["losses"],
        "hits": state["hits"],
        "seen": state["seen"],
        "elapsed": state["elapsed"],
        "val_loss": state.get("val_loss"),
        "val_windows": state.get("val_windows", 0),
        "val_starts": state.get("val_starts", []),
        "val_scored_at": state.get("val_scored_at"),
        "parameters": count_parameters(state["model"]),
        "rng": _dump_rng(state["rng"]),
        "weights": export_weights(state["model"]),
    }


def save_recorded_run(state, path):
    path = Path(path)
    payload = recorded_payload(state)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def load_recorded_run(path, corpus, source):
    """Rebuild the model and copy the recorded weights. Metrics come from the file, not from a new run."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    state = initial_state(corpus, source, config)
    if list(state["fingerprint"]) != list(payload["fingerprint"]):
        raise ValueError("The recorded run does not match this corpus and configuration.")
    if state["stoi"] != {word: int(index) for word, index in payload["stoi"].items()}:
        raise ValueError("The recorded vocabulary does not match this corpus.")
    apply_weights(state["model"], payload["weights"])
    state["parameters"] = count_parameters(state["model"])
    if state["parameters"] != int(payload["parameters"]):
        raise ValueError("The live parameter count does not match the recorded run.")
    state["losses"] = [float(value) for value in payload["losses"]]
    state["hits"] = int(payload["hits"])
    state["seen"] = int(payload["seen"])
    state["steps_done"] = int(payload["steps_done"])
    state["target"] = int(payload["target"])
    state["elapsed"] = float(payload["elapsed"])
    state["lr"] = float(config["lr"])
    state["val_loss"] = None if payload.get("val_loss") is None else float(payload["val_loss"])
    state["val_windows"] = int(payload.get("val_windows") or 0)
    state["val_starts"] = [int(value) for value in payload.get("val_starts") or []]
    scored = payload.get("val_scored_at")
    state["val_scored_at"] = None if scored is None else int(scored)
    state["rng"] = _load_rng(payload["rng"])
    state["running"] = False
    return state


def encode_words(state, prompt):
    words = prompt.split()
    if not words:
        raise ValueError("Enter a token sequence.")
    missing = sorted({word for word in words if word not in state["stoi"]})
    if missing:
        raise ValueError("These words are not in the vocabulary: " + ", ".join(missing))
    ids = [state["stoi"][word] for word in words]
    used = list(words)
    truncated = False
    if len(ids) > state["block_size"]:
        ids = ids[-state["block_size"] :]
        used = used[-state["block_size"] :]
        truncated = True
    return ids, used, truncated
