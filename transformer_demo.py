"""Resume demo for the from-scratch Transformer.

This file only imports the library. It does not modify tensor.py,
mathematics.py, layer.py, or Transformer.py.
"""

import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from Transformer import CrossEntropyLoss, Transformer


DEFAULT_SENTENCE = "The energy of motion is called kinetic energy."


def visible(char):
    if char == " ":
        return "·"
    if char == "\n":
        return "↵"
    return char


def build_vocab(sentence):
    chars = sorted(set(sentence))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for index, char in enumerate(chars)}
    ids = [stoi[char] for char in sentence]
    return chars, stoi, itos, ids


def one_hot(ids, vocab_size):
    rows = []
    for token in ids:
        row = [0.0] * vocab_size
        row[token] = 1.0
        rows.append(row)
    return rows


def argmax_row(values):
    best = 0
    for index, value in enumerate(values):
        if value > values[best]:
            best = index
    return best


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


def matrix_rows(matrix):
    return [[float(value) for value in row.to_list()] for row in matrix]


def capture_attention(model):
    captured = []
    for block in model.blocks:
        heads = []
        for head in block.attention.heads:
            heads.append(matrix_rows(head.sm.predicted))
        captured.append(heads)
    return captured


def future_attention(attention):
    largest = 0.0
    for block in attention:
        for head in block:
            for row_index, row in enumerate(head):
                for column_index, value in enumerate(row):
                    if column_index > row_index:
                        largest = max(largest, value)
    return largest


def run_demo(settings, on_step=None):
    sentence = settings["sentence"]
    window = settings["window"]
    if len(sentence) < window + 1:
        raise ValueError("The sentence needs at least one character after the training window.")

    np.random.seed(settings["seed"])
    chars, stoi, itos, ids = build_vocab(sentence)
    vocab_size = len(chars)
    block_size = window
    model = Transformer(
        vocab_size,
        settings["n_embed"],
        settings["num_heads"],
        block_size,
        settings["n_blocks"],
    )
    loss_fn = CrossEntropyLoss()
    context = ids[:window]
    targets = ids[1 : window + 1]
    target_matrix = one_hot(targets, vocab_size)

    losses = []
    started = time.perf_counter()
    for step in range(settings["steps"]):
        probabilities = model.forward(context)
        loss = float(loss_fn.forward(probabilities, target_matrix))
        model.backward(loss_fn.backward(), settings["lr"])
        losses.append(loss)
        if on_step is not None:
            on_step(step + 1, settings["steps"], loss)
    elapsed = time.perf_counter() - started

    probabilities = model.sm.predicted
    predictions = [argmax_row(probabilities[i].to_list()) for i in range(len(probabilities))]
    attention = capture_attention(model)
    head_size = model.blocks[0].attention.heads[0].w_values.output_dim

    return {
        "sentence": sentence,
        "window": window,
        "steps": settings["steps"],
        "lr": settings["lr"],
        "seed": settings["seed"],
        "n_embed": settings["n_embed"],
        "num_heads": settings["num_heads"],
        "n_blocks": settings["n_blocks"],
        "head_size": head_size,
        "vocab_size": vocab_size,
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
        "block_size": block_size,
        "losses": losses,
        "seconds": elapsed,
        "parameters": count_parameters(model),
        "labels": [sentence[i] for i in range(window)],
        "target_text": "".join(itos[token] for token in targets),
        "predicted_text": "".join(itos[token] for token in predictions),
        "attention": attention,
        "future_mass": future_attention(attention),
        "embedding_shape": model.embedding.output.shape,
        "probability_shape": model.sm.predicted.shape,
        "uniform_loss": math.log(vocab_size),
        "model": model,
    }


def continue_text(model, prompt, n_new, stoi, itos, block_size):
    missing = sorted({char for char in prompt if char not in stoi})
    if not prompt:
        raise ValueError("Enter a prompt made from characters in the training sentence.")
    if missing:
        raise ValueError("These characters were not in the training sentence: " + " ".join(missing))
    context = [stoi[char] for char in prompt]
    for _ in range(n_new):
        window = context[-block_size:]
        probabilities = model.forward(window)
        last = probabilities[len(window) - 1].to_list()
        context.append(argmax_row(last))
    return "".join(itos[token] for token in context)


def show_text(label, text):
    shown = "".join(visible(char) for char in text)
    st.markdown(
        f"<div class='tape'><span>{label}</span><code>{shown}</code></div>",
        unsafe_allow_html=True,
    )


def loss_figure(demo):
    figure, axis = plt.subplots(figsize=(8.2, 3.3))
    axis.plot(range(1, len(demo["losses"]) + 1), demo["losses"], color="#9a3412", linewidth=2.2)
    axis.axhline(demo["uniform_loss"], color="#78716c", linestyle="--", linewidth=1)
    axis.set_xlabel("Step")
    axis.set_ylabel("Cross-entropy")
    axis.set_title("Same window, repeated")
    figure.tight_layout()
    return figure


def attention_figure(weights, labels):
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    image = axis.imshow(weights, cmap="YlOrBr", vmin=0, vmax=1, origin="upper")
    ticks = [visible(char) for char in labels]
    axis.set_xticks(range(len(ticks)), ticks, fontsize=8)
    axis.set_yticks(range(len(ticks)), ticks, fontsize=8)
    axis.set_xlabel("Key")
    axis.set_ylabel("Query")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def inject_style():
    st.markdown(
        """
        <style>
        .stApp { background: #f4f0e6; }
        header[data-testid="stHeader"] { background: transparent; }
        h1, h2, h3 { letter-spacing: -0.03em; }
        .hero-kicker {
            color: #9a3412; font-size: 0.82rem; font-weight: 700;
            letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.2rem;
        }
        .hero-sub { color: #44403c; font-size: 1.05rem; max-width: 46rem; }
        .flow { display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: stretch; margin: 0.6rem 0 1rem; }
        .box {
            background: #fbf8f1; border: 1px solid #e4dccb; border-radius: 10px;
            padding: 0.55rem 0.7rem; min-width: 7.2rem;
        }
        .box b { display: block; color: #1c1915; }
        .box span { color: #78716c; font-family: ui-monospace, monospace; font-size: 0.78rem; }
        .tape {
            display: flex; gap: 0.8rem; align-items: baseline;
            background: #fbf8f1; border: 1px solid #e4dccb; border-radius: 8px;
            padding: 0.45rem 0.7rem; margin-bottom: 0.35rem;
        }
        .tape span { width: 5.5rem; color: #9a3412; font-size: 0.78rem; font-weight: 700; }
        .tape code { color: #1c1915; font-size: 0.95rem; }
        div[data-testid="stMetric"] {
            background: #fbf8f1; border: 1px solid #e4dccb; border-radius: 10px; padding: 0.4rem 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar():
    st.sidebar.header("Demo setup")
    sentence = st.sidebar.text_area("Training sentence", DEFAULT_SENTENCE, height=90)
    longest = max(8, len(sentence) - 1)
    window = st.sidebar.slider("Window length", 8, min(32, longest), min(16, longest))
    steps = st.sidebar.slider("Steps on that window", 10, 40, 36)
    lr = st.sidebar.select_slider("Learning rate", options=[0.2, 0.5, 0.8], value=0.5)
    n_embed = st.sidebar.selectbox("Embedding width", [8, 16], index=1)
    num_heads = st.sidebar.selectbox("Heads", [2, 4], index=1)
    n_blocks = st.sidebar.selectbox("Blocks", [1, 2], index=0)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=9999, value=0, step=1)
    train = st.sidebar.button("Train again", type="primary", use_container_width=True)
    st.sidebar.caption("The first visit trains once. Train again after changing the sentence or the model size.")
    if n_embed % num_heads != 0:
        st.sidebar.error("The embedding width has to be divisible by the number of heads.")
    return {
        "sentence": sentence,
        "window": int(window),
        "steps": int(steps),
        "lr": float(lr),
        "n_embed": int(n_embed),
        "num_heads": int(num_heads),
        "n_blocks": int(n_blocks),
        "seed": int(seed),
        "train": train,
        "valid": n_embed % num_heads == 0 and len(sentence) >= int(window) + 1,
    }


def render_overview(demo):
    st.markdown(
        """
        <div class="flow">
          <div class="box"><b>Token ids</b><span>(T,)</span></div>
          <div class="box"><b>Embed + position</b><span>(T, C)</span></div>
          <div class="box"><b>Pre-norm block</b><span>attention, then MLP</span></div>
          <div class="box"><b>Linear + softmax</b><span>(T, vocab)</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("What is actually in the forward pass")
        st.markdown(
            """
            Each block is the pre-norm residual stack in `TransformerBlock`:

            `out = x + Attention(LayerNorm(x))`
            `out = out + FeedForward(LayerNorm(out))`

            Attention builds queries, keys, and values with `Linear`, scores them with
            `Q @ Kᵀ / sqrt(head size)`, then `Matrix.tril` writes `-inf` above the
            diagonal before the row softmax. Heads are concatenated and mixed with
            another `Linear`. Nothing here calls a framework autograd.
            """
        )
    with right:
        st.subheader("Shapes from the last forward")
        st.write(
            {
                "token embedding": demo["embedding_shape"],
                "softmax rows": demo["probability_shape"],
                "head size": demo["head_size"],
                "parameters": f"{demo['parameters']:,}",
            }
        )
        st.caption("Parameters are counted from the live Matrix and Vector objects, including both embedding tables and every layer norm.")


def render_training(demo):
    start, end, uniform, seconds = st.columns(4)
    start.metric("Start loss", f"{demo['losses'][0]:.3f}")
    end.metric("End loss", f"{demo['losses'][-1]:.3f}")
    uniform.metric("log(vocab)", f"{demo['uniform_loss']:.3f}")
    seconds.metric("Train time", f"{demo['seconds']:.2f}s")

    figure = loss_figure(demo)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.caption("The dashed line is the loss of a uniform guess. Each step uses the same character window, so the curve is the model memorizing that window.")

    matched = sum(a == b for a, b in zip(demo["predicted_text"], demo["target_text"]))
    st.write(f"Greedy next-character reads match the target in {matched} of {len(demo['target_text'])} positions.")
    show_text("Context", "".join(demo["labels"]))
    show_text("Predicts", demo["predicted_text"])
    show_text("Target", demo["target_text"])
    st.caption("A dot stands for a space. Each prediction is the largest probability at that position, and the target is the following character.")

    with st.expander("The training step this button runs"):
        st.code(
            "probs = model.forward(ids[:window])\n"
            "loss = loss_fn.forward(probs, one_hot_next_chars)\n"
            "model.backward(loss_fn.backward(), learning_rate)",
            language="python",
        )
        st.markdown(
            "`CrossEntropyLoss.backward` returns `(p - y) / T`, the gradient with respect to the logits. "
            "`Transformer.backward` starts at the unembedding and walks the blocks from last to first."
        )


def render_attention(demo):
    largest = demo["future_mass"]
    st.write(
        f"Largest attention weight on a future token: **{largest:.2e}**. "
        "A query at position i can only use keys at positions `0 … i`."
    )
    labels = demo["labels"]
    for block_index, heads in enumerate(demo["attention"]):
        st.subheader(f"Block {block_index + 1}")
        for start in range(0, len(heads), 2):
            columns = st.columns(2)
            for offset, weights in enumerate(heads[start : start + 2]):
                with columns[offset]:
                    st.caption(f"Head {start + offset + 1}")
                    figure = attention_figure(weights, labels)
                    st.pyplot(figure, clear_figure=True)
                    plt.close(figure)
    st.caption("Rows are queries and columns are keys, read off `SingleHeadAttention.sm.predicted` after the real forward pass. Darker means more weight.")


def render_generate(demo):
    st.write(
        "Generation takes the last probability row, appends that character, and runs the model again. "
        "The context is cropped to the position table."
    )
    prompt = st.text_input("Prompt", demo["sentence"][:1])
    n_new = st.slider("New characters", 1, demo["window"], demo["window"])
    if st.button("Generate", type="primary"):
        try:
            text = continue_text(
                demo["model"], prompt, int(n_new), demo["stoi"], demo["itos"], demo["block_size"]
            )
            st.session_state.generation = text
        except ValueError as error:
            st.session_state.generation = None
            st.error(str(error))
    generated = st.session_state.get("generation")
    if generated:
        reference = demo["sentence"][: len(generated)]
        matched = sum(a == b for a, b in zip(generated, reference))
        show_text("Output", generated)
        show_text("Source", reference)
        st.caption(f"{matched} of {len(generated)} characters agree with the training sentence, from the start.")


def main():
    st.set_page_config(page_title="Transformer from Scratch", layout="wide")
    inject_style()
    st.markdown("<div class='hero-kicker'>PhysicsLLM · from scratch</div>", unsafe_allow_html=True)
    st.title("Transformer from Scratch")
    st.markdown(
        "<p class='hero-sub'>A decoder-only language model whose weights, attention, and gradients "
        "are Vector and Matrix operations in pure Python. This page trains that code and reads the "
        "attention weights back out of it.</p>",
        unsafe_allow_html=True,
    )

    settings = sidebar()
    needs_train = "demo" not in st.session_state or settings["train"]
    if needs_train:
        if not settings["valid"]:
            st.error("Choose a sentence longer than the window, and an embedding width divisible by the head count.")
            st.stop()
        progress = st.progress(0.0, text="Training the character window")

        def on_step(step, total, loss):
            progress.progress(step / total, text=f"Step {step} of {total} · loss {loss:.3f}")

        try:
            st.session_state.demo = run_demo(settings, on_step)
            st.session_state.generation = None
        except Exception as error:
            st.exception(error)
            st.stop()
        finally:
            progress.empty()

    demo = st.session_state.demo
    stale = (
        demo["sentence"] != settings["sentence"]
        or demo["window"] != settings["window"]
        or demo["n_embed"] != settings["n_embed"]
        or demo["num_heads"] != settings["num_heads"]
        or demo["n_blocks"] != settings["n_blocks"]
        or demo["steps"] != settings["steps"]
        or demo["lr"] != settings["lr"]
        or demo["seed"] != settings["seed"]
    )
    if stale and not settings["train"]:
        st.info("The sidebar no longer matches the trained model. Press Train again to rebuild it.")

    st.caption(
        f"Character vocabulary {demo['vocab_size']} · width {demo['n_embed']} · "
        f"{demo['num_heads']} heads of size {demo['head_size']} · {demo['n_blocks']} block"
        f"{'' if demo['n_blocks'] == 1 else 's'} · {demo['parameters']:,} parameters"
    )

    overview, training, attention, generate = st.tabs(["Architecture", "Training", "Attention", "Generate"])
    with overview:
        render_overview(demo)
    with training:
        render_training(demo)
    with attention:
        render_attention(demo)
    with generate:
        render_generate(demo)


if __name__ == "__main__":
    main()
