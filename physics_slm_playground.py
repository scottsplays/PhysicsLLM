"""Runnable registry for the handwritten classes.

Every runner builds its own instance. Backward passes update that copy and
never a model stored by the training page. This module does not load
dataloader.py.
"""

import math
from pathlib import Path

import numpy as np

import mathematics
from physics_slm_core import causal_report, one_hot, shape_of, to_rows
from tensor import Matrix, Vector
from Transformer import (
    CrossEntropyLoss,
    Embedding,
    FeedForward,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    PositionalEncoding,
    ReLU,
    SingleHeadAttention,
    Softmax,
    Transformer,
    TransformerBlock,
)

ROOT = Path(__file__).resolve().parent
LIBRARY_FILES = ("mathematics.py", "tensor.py", "Transformer.py", "dataloader.py")

MATH_BLURBS = {
    "relu": "Pointwise rectifier, written as x/2 + |x|/2, which is max(x, 0).",
    "drelu": "Helper derivative: 0 for a negative input and 1 otherwise, including at zero. ReLU.backward masks every input that is less than or equal to zero, so the two disagree at exactly 0.",
    "exp": "Natural exponential, NumPy's exp.",
    "log": "Natural log, NumPy's log.",
    "mean": "Arithmetic mean.",
    "var": "Population variance. Squared deviations are divided by N, not by N − 1.",
    "softmax": "Subtract the max, exponentiate, and divide by the sum. The result is a probability vector.",
    "random": "One uniform draw from NumPy.",
    "linear": "Identity activation used by the older multilayer perceptron.",
    "CrossEntropyLoss": "A function of two lists in mathematics.py. The Transformer's loss is the class in Transformer.py, which is a different object.",
}


def read_library(name):
    if name not in LIBRARY_FILES:
        raise ValueError("That file is not part of the library browser.")
    return (ROOT / name).read_text(encoding="utf-8")


def list_symbols(source):
    found = []
    for line in source.splitlines():
        if line.startswith("class ") or line.startswith("def "):
            kind = "class" if line.startswith("class ") else "def"
            ident = line.split()[1].split("(")[0].split(":")[0]
            found.append((kind, ident))
    return found


def extract_symbol(source, name):
    lines = source.splitlines()
    start = None
    for index, line in enumerate(lines):
        if not (line.startswith("class ") or line.startswith("def ")):
            continue
        ident = line.split()[1].split("(")[0].split(":")[0]
        if ident == name:
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("class ") or lines[index].startswith("def "):
            end = index
            break
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end]), start + 1, end


def parse_numbers(text):
    pieces = [piece.strip() for piece in text.replace(";", ",").split(",") if piece.strip()]
    if not pieces:
        raise ValueError("Enter at least one number, separated by commas.")
    values = []
    for piece in pieces:
        try:
            value = float(piece)
        except ValueError as error:
            raise ValueError(f"'{piece}' is not a number.") from error
        if not math.isfinite(value):
            raise ValueError("Every entry has to be a finite number.")
        values.append(value)
    return values


def parse_ids(text):
    ids = []
    for value in parse_numbers(text):
        if abs(value - round(value)) > 1e-9:
            raise ValueError("Token ids have to be whole numbers.")
        if value < 0:
            raise ValueError("Token ids have to be 0 or greater.")
        ids.append(int(round(value)))
    return ids


def parse_matrix(text):
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(parse_numbers(line))
    if not rows:
        raise ValueError("Enter one row of numbers per line.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Every row needs the same number of values.")
    if len(rows) > 16 or width > 32:
        raise ValueError("Use a matrix of at most 16 by 32 in the playground.")
    return Matrix(rows)


def _seed():
    np.random.seed(0)


def _limit_ids(ids):
    if len(ids) > 16:
        raise ValueError("The playground runs at most 16 tokens.")
    if max(ids) + 1 > 64:
        raise ValueError("Keep token ids below 64 in the playground.")
    return ids


def _ones(rows, cols):
    return Matrix([[1.0] * cols for _ in range(rows)])


def _pack(kind, value, details, note):
    rows = None
    text = value
    shape = None
    if isinstance(value, Matrix):
        rows = to_rows(value)
        text = _format(rows)
        shape = value.shape
    elif isinstance(value, Vector):
        rows = [float(item) for item in value.to_list()]
        text = " ".join(f"{item:.4f}" for item in rows)
        shape = (len(rows),)
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        rows = value
        text = _format(rows)
        shape = shape_of(rows)
    elif isinstance(value, (list, tuple)):
        text = ", ".join(f"{float(item):.4f}" for item in value)
        shape = (len(value),)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        text = f"{float(value):.6f}"
        shape = ()
    return {"kind": kind, "shape": shape, "text": text, "details": details, "note": note}


def _format(rows, max_rows=6, max_cols=8):
    shown = []
    for row in rows[:max_rows]:
        cells = []
        for value in row[:max_cols]:
            cells.append(f"{value:.4f}" if math.isfinite(value) else "-inf")
        if len(row) > max_cols:
            cells.append("...")
        shown.append("  ".join(cells))
    if len(rows) > max_rows:
        shown.append("...")
    return "\n".join(shown)


def _run_math(name, values):
    numbers = parse_numbers(values["text"])
    if name == "relu":
        output = [float(mathematics.relu(value)) for value in numbers]
    elif name == "drelu":
        output = [int(mathematics.drelu(value)) for value in numbers]
    elif name == "exp":
        output = [float(mathematics.exp(value)) for value in numbers]
    elif name == "log":
        if any(value <= 0 for value in numbers):
            raise ValueError("log is only defined here for positive numbers.")
        output = [float(mathematics.log(value)) for value in numbers]
    elif name == "mean":
        output = float(mathematics.mean(numbers))
    elif name == "var":
        output = float(mathematics.var(numbers))
    elif name == "softmax":
        output = [float(value) for value in mathematics.softmax(numbers)]
    else:
        raise ValueError(f"{name} is not wired to a runner.")
    return _pack("fixed", output, [], "No weights. The function is the one in mathematics.py.")


def _run_vector(values):
    left = Vector(parse_numbers(values["left"]))
    right = Vector(parse_numbers(values["right"]))
    op = values["op"]
    if op == "add":
        output = left + right
    elif op == "subtract":
        output = left - right
    elif op == "dot":
        output = left @ right
    else:
        raise ValueError("Unknown vector operation.")
    return _pack("fixed", output, [("left", str(left.shape)), ("right", str(right.shape))], "Vector arithmetic from tensor.py. No weights.")


def _run_matrix(values):
    left = parse_matrix(values["left"])
    op = values["op"]
    if op == "tril":
        output = left.tril()
        note = "Entries with column > row become -inf. Attention uses this as the causal mask, before softmax."
    elif op == "matrix multiply":
        right = parse_matrix(values["right"])
        output = left @ right
        note = "Matrix multiply from tensor.py. No weights."
    else:
        raise ValueError("Unknown matrix operation.")
    return _pack("fixed", output, [("left", str(left.shape))], note)


def _embedding(values, backward):
    ids = _limit_ids(parse_ids(values["text"]))
    vocab = max(ids) + 1
    embed = 4
    _seed()
    layer = Embedding(vocab, embed)
    before = list(layer.embedding_table[ids[0]].to_list())
    output = layer.forward(ids)
    if not backward:
        return _pack(
            "random",
            output,
            [("table", f"({vocab}, {embed})"), ("ids", str(ids))],
            "Random embedding table, seed 0. The output rows are copies of the looked-up table rows.",
        )
    dy = _ones(len(ids), embed)
    returned = layer.backward(dy, 0.1)
    after = list(layer.embedding_table[ids[0]].to_list())
    return _pack(
        "random",
        output,
        [
            ("backward return", str(returned)),
            ("row 0 before", " ".join(f"{value:.4f}" for value in before)),
            ("row 0 after", " ".join(f"{value:.4f}" for value in after)),
        ],
        "backward returns None and updates only the looked-up rows. This copy is not the trained model.",
    )


def _positions(values, backward):
    length = int(values["length"])
    if length < 1 or length > 16:
        raise ValueError("Use a length from 1 to 16.")
    _seed()
    layer = PositionalEncoding(length, 4)
    output = layer.forward(list(range(length)))
    if not backward:
        return _pack(
            "random",
            output,
            [("table", f"({length}, 4)")],
            "The values in the input are ignored. The encoding uses positions 0 .. length-1, and the table has exactly `length` rows.",
        )
    before = list(layer.pos_emb.embedding_table[0].to_list())
    returned = layer.backward(_ones(length, 4), 0.1)
    after = list(layer.pos_emb.embedding_table[0].to_list())
    return _pack(
        "random",
        output,
        [
            ("backward return", str(returned)),
            ("position 0 before", " ".join(f"{value:.4f}" for value in before)),
            ("position 0 after", " ".join(f"{value:.4f}" for value in after)),
        ],
        "Same update rule as Embedding, applied to position rows. Random table, seed 0.",
    )


def _linear(values, backward):
    matrix = parse_matrix(values["text"])
    _seed()
    layer = Linear(matrix.m, 2)
    output = layer.forward(matrix)
    if not backward:
        return _pack(
            "random",
            output,
            [("weights", str(layer.weights.shape))],
            "Random Linear, seed 0, 2 outputs, no bias. y = x W.",
        )
    dx = layer.backward(_ones(matrix.n, 2), 0.1)
    return _pack(
        "random",
        dx,
        [("dW", str(layer.grad.shape)), ("learning rate", "0.1")],
        "The shown matrix is dx. Weights on this copy were updated with W = W - 0.1 dW.",
    )


def _relu(values, backward):
    matrix = parse_matrix(values["text"])
    layer = ReLU()
    output = layer.forward(matrix)
    if not backward:
        return _pack("fixed", output, [], "ReLU has no weights. Negative inputs become 0.")
    dx = layer.backward(_ones(matrix.n, matrix.m))
    return _pack(
        "fixed",
        dx,
        [],
        "Upstream gradient of 1 at every cell. A pre-activation that is <= 0 contributes 0. This is the class rule, not mathematics.drelu.",
    )


def _softmax(values, backward):
    matrix = parse_matrix(values["text"])
    layer = Softmax()
    output = layer.forward(matrix)
    if not backward:
        sums = [f"{sum(row):.4f}" for row in to_rows(output)]
        return _pack("fixed", output, [("row sums", ", ".join(sums))], "Each row is a probability vector.")
    grad = layer.backward(_ones(matrix.n, matrix.m))
    sums = [sum(row) for row in to_rows(grad)]
    return _pack(
        "fixed",
        grad,
        [("gradient row sums", ", ".join(f"{value:.4e}" for value in sums))],
        "Softmax backward. Each row of the gradient sums to about 0.",
    )


def _layernorm(values, backward):
    matrix = parse_matrix(values["text"])
    layer = LayerNorm(matrix.m)
    output = layer.forward(matrix)
    if not backward:
        return _pack(
            "fixed",
            output,
            [("gamma", str(layer.gamma.shape)), ("beta", str(layer.beta.shape))],
            "gamma starts at 1 and beta at 0, so this forward is a centered and scaled copy of each row.",
        )
    dx = layer.backward(_ones(matrix.n, matrix.m), 0.1)
    return _pack(
        "fixed",
        dx,
        [],
        "dx after one layer-norm update. gamma and beta on this copy moved by learning rate 0.1.",
    )


def _feedforward(values, backward):
    matrix = parse_matrix(values["text"])
    _seed()
    layer = FeedForward(matrix.m)
    output = layer.forward(matrix)
    if not backward:
        return _pack(
            "random",
            output,
            [("hidden", str(layer.l1.weights.shape)), ("output", str(output.shape))],
            "Random feed-forward, seed 0. The hidden width is 2 times the input width, then ReLU, then a projection back.",
        )
    dx = layer.backward(_ones(matrix.n, matrix.m), 0.1)
    return _pack("random", dx, [], "dx from a fresh random feed-forward. That copy's weights were updated.")


def _single(values, backward):
    matrix = parse_matrix(values["text"])
    head_size = min(4, matrix.m)
    _seed()
    layer = SingleHeadAttention(matrix.m, head_size)
    output = layer.forward(matrix)
    report = causal_report(to_rows(layer.sm.predicted))
    if not backward:
        return _pack(
            "random",
            output,
            [
                ("Q", str(layer.queries.shape)),
                ("K", str(layer.keys.shape)),
                ("V", str(layer.values.shape)),
                ("scores", str(layer.wei.shape)),
                ("probabilities", str(layer.sm.predicted.shape)),
                ("max future probability", f"{report['max_future']:.3e}"),
            ],
            "Random head, seed 0. Scores are Q K^T times 1/sqrt(head size), then tril writes -inf above the diagonal, then softmax.",
        )
    dx = layer.backward(_ones(matrix.n, head_size), 0.1)
    return _pack(
        "random",
        dx,
        [("dx", str(dx.shape))],
        "Backward on a fresh random head. The gradient is not passed through tril; masked probabilities are already 0.",
    )


def _multi(values, backward):
    matrix = parse_matrix(values["text"])
    if matrix.m % 2 != 0:
        raise ValueError("Multi-head demo uses 2 heads, so the row width has to be even.")
    _seed()
    layer = MultiHeadAttention(matrix.m, 2, matrix.m // 2)
    output = layer.forward(matrix)
    if not backward:
        return _pack(
            "random",
            output,
            [("heads", "2"), ("projected", str(output.shape))],
            "Random multi-head, seed 0. Head outputs are concatenated and projected with a Linear of shape (width, width).",
        )
    dx = layer.backward(_ones(matrix.n, matrix.m), 0.1)
    return _pack("random", dx, [("dx", str(dx.shape))], "Backward on a fresh random multi-head. The projection and every head were updated on this copy.")


def _block(values, backward):
    matrix = parse_matrix(values["text"])
    if matrix.m % 2 != 0:
        raise ValueError("The block demo uses 2 heads, so the row width has to be even.")
    _seed()
    layer = TransformerBlock(matrix.m, 2)
    output = layer.forward(matrix)
    if not backward:
        return _pack(
            "random",
            output,
            [("output", str(output.shape))],
            "Pre-norm block, seed 0: x + attention(layer_norm(x)), then that result plus feed_forward(layer_norm(result)).",
        )
    dx = layer.backward(_ones(matrix.n, matrix.m), 0.1)
    return _pack("random", dx, [("dx", str(dx.shape))], "Full block backward on a fresh random block, learning rate 0.1.")


def _transformer(values, backward):
    ids = _limit_ids(parse_ids(values["text"]))
    vocab = max(ids) + 1
    block = max(4, len(ids))
    _seed()
    model = Transformer(vocab, 8, 2, block, 1)
    probs = model.forward(ids)
    last = [float(value) for value in probs[len(ids) - 1].to_list()]
    if not backward:
        return _pack(
            "random",
            probs,
            [
                ("probabilities", str(probs.shape)),
                ("last-row argmax", str(int(np.argmax(last)))),
                ("parameters", str(_live_count(model))),
            ],
            "Random Transformer, seed 0: width 8, 2 heads, 1 block. The forward return is already softmax probabilities, not logits.",
        )
    targets = ids[1:] + [ids[0]]
    loss_fn = CrossEntropyLoss()
    before = float(loss_fn.forward(probs, one_hot(targets, vocab)))
    returned = model.backward(loss_fn.backward(), 0.5)
    after = float(loss_fn.forward(model.forward(ids), one_hot(targets, vocab)))
    return _pack(
        "random",
        probs,
        [
            ("backward return", str(returned)),
            ("loss before", f"{before:.4f}"),
            ("loss after one step", f"{after:.4f}"),
            ("targets", str(targets)),
        ],
        "One toy update at learning rate 0.5. Targets are the ids rotated by one. This copy started from seed 0 and is not the trained model. One step is not a measured training curve.",
    )


def _live_count(model):
    from physics_slm_core import count_parameters

    return count_parameters(model)


def _cross_entropy(values, backward):
    probs = parse_matrix(values["text"])
    ids = parse_ids(values["targets"])
    if len(ids) != probs.n:
        raise ValueError("Need one class id for each probability row.")
    if any(token >= probs.m for token in ids):
        raise ValueError("A class id is outside the probability row.")
    loss_fn = CrossEntropyLoss()
    loss = float(loss_fn.forward(probs, one_hot(ids, probs.m)))
    if not backward:
        return _pack(
            "fixed",
            loss,
            [("rows", str(probs.n)), ("classes", str(probs.m))],
            "Mean over rows of -log p(correct class). The model returns probabilities, and this loss expects those probabilities.",
        )
    grad = loss_fn.backward()
    return _pack(
        "fixed",
        grad,
        [("loss", f"{loss:.6f}")],
        "Gradient (p - y) / T. Transformer.backward consumes this and does not run softmax backward again.",
    )


def component_names():
    return [spec["name"] for spec in SPECS]


def spec_for(name):
    for spec in SPECS:
        if spec["name"] == name:
            return spec
    return None


def run_component(name, values=None, backward=False):
    spec = spec_for(name)
    if spec is None or spec.get("runner") is None:
        raise ValueError(f"{name} does not have a playground runner.")
    if values is None:
        values = {field["key"]: field["default"] for field in spec["fields"]}
    if backward and not spec.get("backward"):
        raise ValueError(f"{name} has no backward demonstration.")
    return spec["runner"](values, backward) if spec.get("passes_backward") else (
        spec["runner"](values) if not backward else spec["backward_runner"](values)
    )


def _math_runner(name):
    def run(values, backward=False):
        if backward:
            raise ValueError(f"{name} is a plain function and has no backward runner here.")
        return _run_math(name, values)

    return run


def _vector_runner(values, backward=False):
    if backward:
        raise ValueError("Vector arithmetic has no backward runner.")
    return _run_vector(values)


def _matrix_runner(values, backward=False):
    if backward:
        raise ValueError("These matrix operations have no backward runner.")
    return _run_matrix(values)


def _field_text(label, default, kind="area", height=120):
    return {"key": "text", "kind": kind, "label": label, "default": default, "height": height}


SPECS = [
    {
        "name": "relu",
        "file": "mathematics.py",
        "symbol": "relu",
        "summary": MATH_BLURBS["relu"],
        "equation": r"\mathrm{relu}(x)=\max(x,0)",
        "input_shape": "a list of numbers",
        "output_shape": "the same length",
        "fields": [_field_text("Numbers", "-1, 0, 2, 4", "text")],
        "backward": False,
        "passes_backward": True,
        "runner": _math_runner("relu"),
    },
    {
        "name": "drelu",
        "file": "mathematics.py",
        "symbol": "drelu",
        "summary": MATH_BLURBS["drelu"],
        "equation": r"d\mathrm{relu}(x)=0 \text{ if } x<0 \text{ else } 1",
        "input_shape": "a list of numbers",
        "output_shape": "the same length",
        "fields": [_field_text("Numbers", "-1, 0, 2", "text")],
        "backward": False,
        "passes_backward": True,
        "runner": _math_runner("drelu"),
    },
    {
        "name": "softmax",
        "file": "mathematics.py",
        "symbol": "softmax",
        "summary": MATH_BLURBS["softmax"],
        "equation": r"\mathrm{softmax}(x)_i=\exp(x_i-\max x)/\sum_j\exp(x_j-\max x)",
        "input_shape": "one vector",
        "output_shape": "one probability vector",
        "fields": [_field_text("Numbers", "1, 2, 3", "text")],
        "backward": False,
        "passes_backward": True,
        "runner": _math_runner("softmax"),
    },
    {
        "name": "mean",
        "file": "mathematics.py",
        "symbol": "mean",
        "summary": MATH_BLURBS["mean"],
        "equation": r"\bar x = \frac{1}{N}\sum_i x_i",
        "input_shape": "a list of numbers",
        "output_shape": "a scalar",
        "fields": [_field_text("Numbers", "1, 2, 3, 4", "text")],
        "backward": False,
        "passes_backward": True,
        "runner": _math_runner("mean"),
    },
    {
        "name": "var",
        "file": "mathematics.py",
        "symbol": "var",
        "summary": MATH_BLURBS["var"],
        "equation": r"\mathrm{var}(x)=\frac{1}{N}\sum_i (x_i-\bar x)^2",
        "input_shape": "a list of numbers",
        "output_shape": "a scalar",
        "fields": [_field_text("Numbers", "1, 2, 3, 4", "text")],
        "backward": False,
        "passes_backward": True,
        "runner": _math_runner("var"),
    },
    {
        "name": "exp",
        "file": "mathematics.py",
        "symbol": "exp",
        "summary": MATH_BLURBS["exp"],
        "equation": r"\exp(x)",
        "input_shape": "a list of numbers",
        "output_shape": "the same length",
        "fields": [_field_text("Numbers", "0, 1", "text")],
        "backward": False,
        "passes_backward": True,
        "runner": _math_runner("exp"),
    },
    {
        "name": "log",
        "file": "mathematics.py",
        "symbol": "log",
        "summary": MATH_BLURBS["log"],
        "equation": r"\log(x)",
        "input_shape": "a list of positive numbers",
        "output_shape": "the same length",
        "fields": [_field_text("Numbers", "1, 2", "text")],
        "backward": False,
        "passes_backward": True,
        "runner": _math_runner("log"),
    },
    {
        "name": "Vector",
        "file": "tensor.py",
        "symbol": "Vector",
        "summary": "A 1-D list of floats with addition, subtraction, and a dot product. Matrices are built from these rows.",
        "equation": r"a\cdot b=\sum_i a_i b_i",
        "input_shape": "(n,) and (n,)",
        "output_shape": "(n,) or a scalar for the dot product",
        "fields": [
            {"key": "op", "kind": "choice", "label": "Operation", "choices": ["add", "subtract", "dot"], "default": "add"},
            {"key": "left", "kind": "text", "label": "Left", "default": "1, 2, 3"},
            {"key": "right", "kind": "text", "label": "Right", "default": "4, 5, 6"},
        ],
        "backward": False,
        "passes_backward": True,
        "runner": _vector_runner,
    },
    {
        "name": "Matrix",
        "file": "tensor.py",
        "symbol": "Matrix",
        "summary": "A 2-D table of Vectors. matmul is the product used everywhere in the Transformer. tril is the causal mask.",
        "equation": r"(AB)_{ij}=\sum_k A_{ik} B_{kj}",
        "input_shape": "(n, k) and (k, m), or one square-or-rectangular matrix for tril",
        "output_shape": "(n, m)",
        "fields": [
            {"key": "op", "kind": "choice", "label": "Operation", "choices": ["matrix multiply", "tril"], "default": "matrix multiply"},
            {"key": "left", "kind": "area", "label": "Left", "default": "1, 2\n3, 4", "height": 90},
            {"key": "right", "kind": "area", "label": "Right", "default": "5, 6\n7, 8", "height": 90, "hide_when": ("op", ["tril"])},
        ],
        "backward": False,
        "passes_backward": True,
        "runner": _matrix_runner,
    },
    {
        "name": "Embedding",
        "file": "Transformer.py",
        "symbol": "Embedding",
        "summary": "Looks up a row of the embedding table for each token id. The table is a Matrix of shape (vocabulary, width), not a Linear layer.",
        "equation": r"X_t = E_{i_t}",
        "input_shape": "list of T integer ids",
        "output_shape": "(T, embed)",
        "fields": [_field_text("Token ids", "0, 2, 1", "text")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "token_embeddings",
        "runner": _embedding,
    },
    {
        "name": "PositionalEncoding",
        "file": "Transformer.py",
        "symbol": "PositionalEncoding",
        "summary": "An embedding table with one row per position. forward uses range(length), not the token ids. The highest legal position is block_size - 1.",
        "equation": r"P_t = E^{\mathrm{pos}}_{t}",
        "input_shape": "a length T, with T <= table rows",
        "output_shape": "(T, embed)",
        "fields": [{"key": "length", "kind": "number", "label": "Sequence length", "default": 4, "min": 1, "max": 16, "step": 1}],
        "backward": True,
        "passes_backward": True,
        "trained_key": "position_embeddings",
        "runner": _positions,
    },
    {
        "name": "Linear",
        "file": "Transformer.py",
        "symbol": "Linear",
        "summary": "A bias-free matrix multiply. Attention projections, the feed-forward layers, and the unembedding are all this class.",
        "equation": r"Y = X W",
        "input_shape": "(T, in)",
        "output_shape": "(T, out)",
        "fields": [_field_text("Rows of x", "1, 0, -1\n0.5, 2, 0")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "logits",
        "runner": _linear,
    },
    {
        "name": "ReLU",
        "file": "Transformer.py",
        "symbol": "ReLU",
        "summary": "The feed-forward nonlinearity. Backward reads the pre-activation, and a value <= 0 blocks the gradient.",
        "equation": r"\mathrm{ReLU}(x)=\max(x,0)",
        "input_shape": "(T, C)",
        "output_shape": "(T, C)",
        "fields": [_field_text("Rows", "-1, 0, 2\n0.5, -2, 4")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "relu",
        "runner": _relu,
    },
    {
        "name": "FeedForward",
        "file": "Transformer.py",
        "symbol": "FeedForward",
        "summary": "Linear to 2 times the width, ReLU, then Linear back to the width. Applied at every position with the same weights.",
        "equation": r"\mathrm{FF}(X)=\mathrm{ReLU}(X W_1)\,W_2",
        "input_shape": "(T, C)",
        "output_shape": "(T, C)",
        "fields": [_field_text("Rows", "0.2, -0.1, 0.4, 0.0\n-0.3, 0.5, 0.1, -0.2")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "feedforward",
        "runner": _feedforward,
    },
    {
        "name": "Softmax",
        "file": "Transformer.py",
        "symbol": "Softmax",
        "summary": "Row-wise softmax. Attention uses one instance per head. The Transformer uses another after the unembedding, so the model returns probabilities.",
        "equation": r"p_{t,i}=\mathrm{softmax}(z_t)_i",
        "input_shape": "(T, K)",
        "output_shape": "(T, K)",
        "fields": [_field_text("Rows", "1, 2, 3\n0, 0, 1")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "probs",
        "runner": _softmax,
    },
    {
        "name": "LayerNorm",
        "file": "Transformer.py",
        "symbol": "LayerNorm",
        "summary": "Normalize each row to mean 0 and unit second moment, then scale by gamma and shift by beta. Used before attention, before the feed-forward, and once more at the top of the stack.",
        "equation": r"y=(x-\mu)/\sqrt{\mathrm{var}(x)+\epsilon}\cdot\gamma+\beta",
        "input_shape": "(T, C)",
        "output_shape": "(T, C)",
        "fields": [_field_text("Rows", "1, 2, 3, 4\n4, 3, 2, 1")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "final_ln",
        "runner": _layernorm,
    },
    {
        "name": "SingleHeadAttention",
        "file": "Transformer.py",
        "symbol": "SingleHeadAttention",
        "summary": "One causal attention head. Q, K, and V are linear maps of the input. Scores are scaled, future positions are set to -inf, and softmax produces the weights that mix V.",
        "equation": r"\mathrm{softmax}\!\left(\mathrm{tril}(QK^{\top}/\sqrt{d_h})\right)V",
        "input_shape": "(T, n_embed)",
        "output_shape": "(T, head_size)",
        "fields": [_field_text("Rows, width is n_embed", "0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.5\n-0.4, 0.2, 0.1, 0.3, -0.1, 0.2, 0.0, 0.4\n0.1, 0.1, -0.3, 0.2, 0.4, 0.0, -0.2, 0.3")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "head",
        "needs_head": True,
        "runner": _single,
    },
    {
        "name": "MultiHeadAttention",
        "file": "Transformer.py",
        "symbol": "MultiHeadAttention",
        "summary": "Several heads on the same input. Their outputs are concatenated along the feature axis and mixed by a projection. The demo uses 2 heads and requires an even width.",
        "equation": r"\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O",
        "input_shape": "(T, n_embed), with n_embed = heads × head size",
        "output_shape": "(T, n_embed)",
        "fields": [_field_text("Rows, even width", "0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.5\n-0.4, 0.2, 0.1, 0.3, -0.1, 0.2, 0.0, 0.4")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "attention",
        "runner": _multi,
    },
    {
        "name": "TransformerBlock",
        "file": "Transformer.py",
        "symbol": "TransformerBlock",
        "summary": "A pre-norm residual block. The skip connection adds the block input to each sublayer output, so the gradient has a direct path backward.",
        "equation": r"x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x)),\quad x \leftarrow x + \mathrm{FF}(\mathrm{LN}(x))",
        "input_shape": "(T, n_embed), even width in this demo",
        "output_shape": "(T, n_embed)",
        "fields": [_field_text("Rows, even width", "0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.5\n-0.4, 0.2, 0.1, 0.3, -0.1, 0.2, 0.0, 0.4")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "output",
        "runner": _block,
    },
    {
        "name": "Transformer",
        "file": "Transformer.py",
        "symbol": "Transformer",
        "summary": "Decoder-only language model. Token embeddings plus positions, then blocks, a final layer norm, an unembedding, and a softmax. The returned matrix is next-token probabilities at every position.",
        "equation": r"p_t=\mathrm{softmax}(W_U\,\mathrm{LN}(\mathrm{blocks}(E_{i_t}+P_t)))",
        "input_shape": "list of T token ids, T <= block size",
        "output_shape": "(T, vocabulary)",
        "fields": [_field_text("Token ids", "0, 1, 2", "text")],
        "backward": True,
        "passes_backward": True,
        "trained_key": "probs",
        "runner": _transformer,
    },
    {
        "name": "CrossEntropyLoss",
        "file": "Transformer.py",
        "symbol": "CrossEntropyLoss",
        "summary": "Training loss for next-token ids. forward averages -log p(target) over the window. backward returns (p - y) / T and takes no upstream gradient, because the loss sits at the end of the graph.",
        "equation": r"L=-\frac{1}{T}\sum_t \log p_{t,y_t}",
        "input_shape": "probability matrix (T, V) and one class id per row",
        "output_shape": "a scalar; backward returns (T, V)",
        "fields": [
            _field_text("Probability rows", "0.70, 0.20, 0.10\n0.10, 0.20, 0.70"),
            {"key": "targets", "kind": "text", "label": "Class ids, one per row", "default": "0, 2"},
        ],
        "backward": True,
        "passes_backward": True,
        "runner": _cross_entropy,
    },
]


def trained_view(name, trace, block=0, head=0):
    """Pick the activation that corresponds to a class after a real forward."""
    spec = spec_for(name)
    if spec is None or not spec.get("trained_key"):
        raise ValueError(f"{name} is not read off the trained forward.")
    if not trace["blocks"]:
        raise ValueError("This forward did not run a block.")
    block = int(block)
    head = int(head)
    if block < 0 or block >= len(trace["blocks"]):
        raise ValueError("That block is not in the model.")
    chosen = trace["blocks"][block]
    key = spec["trained_key"]
    note = "Taken from the trained model's forward. Weights were not updated."
    details = []
    if key == "head":
        if head < 0 or head >= len(chosen["heads"]):
            raise ValueError("That head is not in the block.")
        item = chosen["heads"][head]
        rows = item["probs"]
        report = causal_report(rows)
        details = [
            ("Q", str(shape_of(item["Q"]))),
            ("K", str(shape_of(item["K"]))),
            ("V", str(shape_of(item["V"]))),
            ("scores", str(shape_of(item["scores"]))),
            ("max future probability", f"{report['max_future']:.3e}"),
        ]
        note = (
            f"Attention probabilities for block {block + 1}, head {head + 1}. "
            "The matrix below is the softmax, not the head's mixed values."
        )
    elif key in {"attention", "feedforward", "relu", "output", "ln1", "ln2", "residual1"}:
        rows = chosen[key]
        details = [("block", str(block + 1))]
    elif key in trace:
        rows = trace[key]
        if name == "Linear":
            note = "These are the unembedding logits from the trained forward, before the final softmax. Attention and feed-forward use other Linear weights."
        elif name == "LayerNorm":
            note = "This is the final layer norm. Each block also has a layer norm before attention and before the feed-forward."
            details = [
                ("final", str(shape_of(trace["final_ln"]))),
                ("block 1 pre-attention", str(shape_of(trace["blocks"][0]["ln1"]))),
                ("block 1 pre-feed-forward", str(shape_of(trace["blocks"][0]["ln2"]))),
            ]
        elif name == "Softmax":
            note = "Final next-token probabilities. Attention softmax is stored on each head and is a different matrix."
        elif name == "Embedding":
            details = [("table", str(trace["embedding_table"]))]
    else:
        raise ValueError(f"No trained activation is registered for {name}.")
    packed = _pack("trained", rows, details, note)
    return packed
