"""PhysicsSLM portfolio app.

The model, the matrix library, and the training step are the handwritten
classes in Transformer.py, tensor.py, and mathematics.py. This file only
presents them. It does not import dataloader.py.
"""

import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from physics_slm_core import (
    CHECKPOINT_NAME,
    RECORDED_RUN_NAME,
    causal_report,
    compatible_prompts,
    encode_words,
    evaluate_validation,
    fingerprint,
    generate,
    generation_matches,
    initial_state,
    load_recorded_run,
    moving_average,
    parameter_count,
    remember_validation,
    score_held_out,
    trace_forward,
    train_steps,
    uniform_loss,
    window_accuracy,
)
from physics_slm_playground import (
    LIBRARY_FILES,
    ROOT,
    SPECS,
    MATH_BLURBS,
    extract_symbol,
    list_symbols,
    read_library,
    run_component,
    trained_view,
)

BURST = 15
DEFAULTS = {
    "block_size": 16,
    "n_embed": 16,
    "num_heads": 4,
    "n_blocks": 3,
    "seed": 0,
    "lr": 0.05,
}
SECTIONS = [
    "Overview",
    "Train & Evaluate",
    "Text Generation",
    "Architecture Explorer",
    "Code & Component Playground",
]
GITHUB = "https://github.com/scottsplays/PhysicsLLM"
BLOCK_KEYS = {"head", "attention", "feedforward", "relu", "output"}

STAGES = [
    ("tokens", "Input tokens", "The prompt is split on whitespace. Each word must already be in the vocabulary; there is no byte-pair encoding."),
    ("token_emb", "Token embeddings", "Embedding looks up a row of a (vocabulary, width) table. The row is the token's learned vector."),
    ("pos_emb", "Positional embeddings", "A second table has one row per context position, not per word. Position 0 is the first token in the window. The table is the context length, 16 rows in the shipped model. A training script that allocates len(corpus) rows still only reads the first 16 positions."),
    ("sum", "Token + position", "The two vectors are added. That sum is the input to the first block."),
    ("block", "Transformer block", "Pre-norm residual block. Attention is causal. The feed-forward is the same at every position."),
    ("final_ln", "Final LayerNorm", "One more layer norm over the width, after the last block."),
    ("logits", "Output projection", "A bias-free Linear maps the width to the vocabulary. These are logits, before the final softmax."),
    ("probs", "Next-token probabilities", "Row-wise softmax. Training reads every row. Generation reads only the last row, appends that token, and repeats."),
]

INNER = [
    ("ln1", "LayerNorm", "LayerNorm", "Normalize the block input, then compute attention. The residual still carries the original input."),
    ("attn", "Causal multi-head attention", "MultiHeadAttention", "Each head forms Q, K, and V, masks future keys, and mixes values. Heads are concatenated and projected."),
    ("res1", "Residual add", "Tensor add", "The attention output is added to the block input. The backward pass adds the incoming gradient to the sublayer gradient."),
    ("ln2", "LayerNorm", "LayerNorm", "A second layer norm, on the residual stream, before the feed-forward."),
    ("ff", "Feed-forward", "FeedForward", "Linear to twice the width, ReLU, then Linear back. No attention here."),
    ("res2", "Residual add", "Tensor add", "The feed-forward output is added to the stream that entered it."),
]


def checkpoint_path():
    return ROOT / "checkpoints" / CHECKPOINT_NAME


def load_default_corpus():
    path = ROOT / "poc-corpus.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip(), "poc-corpus.txt"
    return (
        "force equals mass times acceleration gravity pulls the ball down the ball falls down",
        "built-in sample",
    )


def read_upload(uploaded):
    return uploaded.getvalue().decode("utf-8", errors="replace").strip(), uploaded.name


def inject_style():
    st.markdown(
        """
        <style>
        .stApp { background: #f3f5f8; color: #172033; }
        header[data-testid="stHeader"] { background: transparent; }
        section[data-testid="stSidebar"] {
            background: #eef2f6;
            border-right: 1px solid #d9e0ea;
        }
        .block-container { padding-top: 1.4rem; max-width: 1180px; }
        h1, h2, h3 { letter-spacing: -0.03em; color: #172033; }
        .kicker {
            color: #1d4e89; font-size: 0.78rem; font-weight: 700;
            letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 0.15rem;
        }
        .subtitle { color: #3d4c5c; font-size: 1.08rem; line-height: 1.55; max-width: 44rem; }
        .pipeline, .inner {
            display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: stretch; margin: 0.4rem 0 0.8rem;
        }
        .node {
            background: #ffffff; border: 1px solid #d9e0ea; border-radius: 10px;
            padding: 0.55rem 0.7rem; min-width: 7.4rem; max-width: 12rem;
        }
        .node.on { border-color: #1d4e89; box-shadow: inset 0 0 0 1px #1d4e89; }
        .node b { display: block; color: #172033; font-size: 0.92rem; }
        .node span { color: #5b6b7c; font-size: 0.78rem; }
        .card {
            background: #ffffff; border: 1px solid #d9e0ea; border-radius: 12px;
            padding: 0.8rem 0.95rem; margin-bottom: 0.7rem;
        }
        .card .label { color: #1d4e89; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        .card p { margin: 0.25rem 0 0; color: #172033; font-size: 1.02rem; line-height: 1.45; }
        div[data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #d9e0ea; border-radius: 12px; padding: 0.55rem 0.75rem;
        }
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] { color: #172033; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_heading(title, caption):
    st.markdown('<p class="kicker">PhysicsSLM</p>', unsafe_allow_html=True)
    st.header(title)
    if caption:
        st.caption(caption)


def model_state():
    return st.session_state.get("slm")


def save_checkpoint(state):
    path = checkpoint_path()
    path.parent.mkdir(exist_ok=True)
    payload = {"version": 1, "state": state}
    state["running"] = False
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_checkpoint():
    path = checkpoint_path()
    if not path.exists():
        raise ValueError(f"No checkpoint at {path}.")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError("That file is not a PhysicsSLM checkpoint.")
    state = payload["state"]
    if "model" not in state or "stoi" not in state:
        raise ValueError("The checkpoint does not contain a model.")
    state["running"] = False
    return state


def loss_figure(losses, vocab_size):
    average = moving_average(losses, 20)
    steps = list(range(1, len(losses) + 1))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=steps, y=losses, name="Window loss", line=dict(color="#9aafc6", width=1.4)))
    figure.add_trace(go.Scatter(x=steps, y=average, name="20-step average", line=dict(color="#1d4e89", width=2.4)))
    figure.add_hline(y=uniform_loss(vocab_size), line_dash="dash", line_color="#8aa0b8")
    figure.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
        font=dict(family="Segoe UI, sans-serif", color="#172033"),
        xaxis_title="Step",
        yaxis_title="Cross-entropy",
    )
    return figure


def heatmap(matrix, x_labels, y_labels, title, zmin=None, zmax=None, colorscale="Blues", height=430):
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            hovertemplate="query %{y}<br>key %{x}<br>value %{z:.4f}<extra></extra>",
            colorbar=dict(thickness=12, len=0.75),
        )
    )
    figure.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=8, r=8, t=42, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", size=12, color="#172033"),
    )
    figure.update_yaxes(autorange="reversed", title="query")
    figure.update_xaxes(title="key")
    return figure


def feature_heatmap(rows, y_labels, title):
    if not rows:
        return None
    x_labels = [f"d{index}" for index in range(len(rows[0]))]
    return heatmap(rows, x_labels, y_labels, title, colorscale="RdBu", height=360)


def token_labels(tokens):
    return [f"{token} [{index}]" for index, token in enumerate(tokens)]


def finite_matrix(rows):
    cleaned = []
    for row in rows:
        cleaned.append([None if value == float("-inf") else value for value in row])
    return cleaned


def sidebar():
    st.sidebar.markdown('<p class="kicker">PhysicsSLM</p>', unsafe_allow_html=True)
    st.sidebar.markdown("**Transformer from scratch**")
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    choice = st.sidebar.radio("Section", SECTIONS, key="page", label_visibility="collapsed")
    state = model_state()
    st.sidebar.divider()
    if state is None:
        st.sidebar.info("No model in memory.")
    elif state["steps_done"] == 0:
        st.sidebar.warning("Model initialized, not trained.")
    else:
        st.sidebar.metric("Trained steps", f"{state['steps_done']:,}")
        st.sidebar.caption(f"Last window loss {state['losses'][-1]:.3f}")
        if state.get("running"):
            st.sidebar.caption(f"Target {state['target']:,} steps.")
    if state and state.get("running") and choice != "Train & Evaluate":
        st.sidebar.caption("Training pauses on this page. Open Train & Evaluate to continue.")
    st.sidebar.divider()
    st.sidebar.markdown(f"[GitHub repository]({GITHUB})")
    st.sidebar.caption("Handwritten Vector, Matrix, and Transformer. No PyTorch in the model.")
    return choice


def render_overview():
    st.markdown('<p class="kicker">PhysicsSLM</p>', unsafe_allow_html=True)
    st.title("PhysicsSLM | Transformer From Scratch")
    st.markdown(
        '<p class="subtitle">An interactive exploration of language modeling, attention, and backpropagation — implemented from first principles.</p>',
        unsafe_allow_html=True,
    )
    st.write(
        "PhysicsSLM is a decoder-only language model. Token embeddings, attention, layer norm, "
        "the feed-forward network, and the backward pass are Python classes on top of a small "
        "matrix library. A training step reads one window, scores the next word at every position, "
        "and updates the weights by hand. It is a study model, not a pretrained production LLM."
    )
    st.write(
        "32-word vocabulary, template continuation only — it does not answer questions and does not do math."
    )

    st.subheader("Short demonstration")
    st.markdown(
        "1. **Train & Evaluate** — start from the sample corpus, or upload a text file.\n"
        "2. **Text Generation** — complete a prompt one word at a time.\n"
        "3. **Architecture Explorer** — open a block and a head and read the attention heatmap.\n"
        "4. **Code & Component Playground** — open a class, read it, and run a forward pass."
    )

    pipeline(["sum"])
    st.caption("Decoder-only stack. The block drawn above is repeated `n_blocks` times. Nothing in the diagram is an encoder.")

    state = model_state()
    st.subheader("Configuration")
    if state is None:
        st.info("Train a corpus, or load a checkpoint, to see the live parameter count. The count depends on the vocabulary.")
        if checkpoint_path().exists():
            st.caption(f"A checkpoint is saved at `{checkpoint_path()}`. Load it from Train & Evaluate.")
    else:
        left, middle, right, far = st.columns(4)
        left.metric("Parameters", f"{state['parameters']:,}")
        middle.metric("Vocabulary", f"{state['vocab_size']:,}")
        right.metric("Context", state["block_size"])
        far.metric("Width × heads × blocks", f"{state['n_embed']} × {state['num_heads']} × {state['n_blocks']}")
        formula = parameter_count(state["vocab_size"], state["n_embed"], state["num_heads"], state["block_size"], state["n_blocks"])
        st.caption(
            f"Parameters are counted from the live weight tables ({state['parameters']:,}). "
            f"The same total from the layer shapes is {formula:,}. "
            f"The position table has {state['block_size']} rows. "
            f"Seed {state['seed']}. Source: {state['source']}."
        )
        if state["steps_done"] == 0:
            st.warning("This model has been initialized and has not taken a training step.")
        else:
            _run_metrics(state)

    st.subheader("What is implemented")
    st.markdown(
        "- Custom `Vector` and `Matrix` operations, including multiply and the causal `tril`\n"
        "- Token embeddings and learned positional embeddings\n"
        "- Pre-norm decoder blocks with residual adds\n"
        "- Causal multi-head attention, with the mask applied before softmax\n"
        "- Position-wise feed-forward network and ReLU\n"
        "- Layer norm with a learned gamma and beta\n"
        "- Next-token cross-entropy and a full backward pass\n"
        "- Greedy decoding, plus temperature sampling applied in this app to the model's probabilities"
    )
    st.caption(f"Source: [PhysicsLLM on GitHub]({GITHUB}). The Transformer lives in `Transformer.py`. The earlier Iris multilayer perceptron is still in `iris_app.py`.")

    if state and state["steps_done"] > 0:
        st.subheader("Prompt completion")
        st.caption("Greedy decoding, three new words, from the model currently in memory. Not a benchmark.")
        samples = cached_samples(state)
        if not samples:
            st.info("None of the built-in prompts are in this vocabulary.")
        for sample in samples:
            st.markdown(
                f"<div class='card'><div class='label'>{sample['prompt']}</div><p>{sample['full']}</p></div>",
                unsafe_allow_html=True,
            )

    st.subheader("About this project")
    st.write(
        "I built this to see a Transformer by writing the pieces: the matrix product, the attention scores, "
        "the residual block, and the gradient that updates each weight. The training target is the next word. "
        "For a window of token ids, the model returns a probability row at every position, the loss is the mean "
        "of −log p(next word), and `backward` walks from the unembedding back through the blocks into both embedding tables."
    )
    st.write(
        "On a small repeated corpus the loss can fall below the loss of a uniform guess, and greedy completions "
        "stay inside that vocabulary. The heatmaps are the softmax weights from a real forward pass. "
        "A particular head is not labeled with a semantic job; the picture shows where the mass sits, and the causal mask forces future positions to zero."
    )
    st.write(
        "Validation loss is a forward-only score on fixed windows from the last 10% of the loaded corpus. "
        "The model still only continues templates from its vocabulary."
    )

    st.subheader("Technical limitations")
    st.markdown(
        "- The vocabulary is the set of whitespace-separated words in the training file. An unknown word cannot be prompted or generated.\n"
        "- There is no batch, no GPU, and no BPE. One step is one window of pure-Python multiplies.\n"
        "- The shipped model is narrow (width 16, four heads, three blocks, context 16). It is not a production language model.\n"
        "- The loss curve is the loss of random training windows from the first 90% of tokens. It moves around even while the average falls. The held-out number is the separate tail score.\n"
        "- Sampled-window accuracy uses those same training windows. It is not a held-out score.\n"
        "- Temperature sampling lives in the app. The model itself always returns a softmax row; training in `dataloader.py` picks the argmax.\n"
        "- `dataloader.py` trains as soon as it is imported, so this app reads that file as text and does not import it.\n"
        "- Checkpoints are a local pickle of this process's model. They are a demo convenience, not a portable format."
    )


def pipeline(selected):
    parts = [
        ("tokens", "Input tokens", "ids"),
        ("token_emb", "Token embeddings", "Embedding"),
        ("pos_emb", "Positions", "PositionalEncoding"),
        ("sum", "Add", "token + position"),
        ("block", "Transformer blocks", "× n_blocks"),
        ("final_ln", "Final LayerNorm", "LayerNorm"),
        ("logits", "Output projection", "Linear"),
        ("probs", "Softmax", "next-token p"),
    ]
    html = ["<div class='pipeline'>"]
    for key, title, detail in parts:
        mark = " on" if key in selected else ""
        html.append(f"<div class='node{mark}'><b>{title}</b><span>{detail}</span></div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def inner_pipeline(selected):
    html = ["<div class='inner'>"]
    for key, title, detail in (
        ("ln1", "LayerNorm", "pre-attention"),
        ("attn", "Multi-head attention", "causal"),
        ("res1", "Residual", "+ input"),
        ("ln2", "LayerNorm", "pre-FFN"),
        ("ff", "Feed-forward", "ReLU"),
        ("res2", "Residual", "+ stream"),
    ):
        mark = " on" if key == selected else ""
        html.append(f"<div class='node{mark}'><b>{title}</b><span>{detail}</span></div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def cached_samples(state):
    if state.get("sample_step") == state["steps_done"] and state.get("samples"):
        return state["samples"]
    samples = []
    for prompt in compatible_prompts(state["stoi"])[:3]:
        samples.append(generate(state["model"], state["stoi"], state["itos"], prompt, 3, state["block_size"], 0.0))
    state["samples"] = samples
    state["sample_step"] = state["steps_done"]
    return samples


def render_train():
    page_heading("Train & Evaluate", "One step is one random window. The weights are the handwritten Transformer, updated by its own backward pass.")
    uploaded = st.file_uploader("Corpus file", type=["txt"])
    if uploaded is not None:
        corpus, source = read_upload(uploaded)
    else:
        corpus, source = load_default_corpus()
        st.caption(f"No file uploaded. Using {source} until you choose one.")
    if not corpus:
        st.error("The corpus is empty.")
        return
    words = corpus.split()
    distinct = len(set(words))
    preview = " ".join(words[:24])
    c1, c2, c3 = st.columns(3)
    c1.metric("Words", f"{len(words):,}")
    c2.metric("Vocabulary", f"{distinct:,}")
    c3.metric("Preview words", min(24, len(words)))
    if preview:
        st.caption(preview + (" …" if len(words) > 24 else ""))
    if distinct > 200:
        st.warning("This vocabulary is large for pure-Python multiplies. A step may be slow. The sample corpus stays small on purpose.")

    left, right = st.columns(2)
    with left:
        steps = st.number_input("Steps", min_value=10, max_value=5000, value=1000, step=10, key="train_steps")
    with right:
        lr = st.number_input("Learning rate", min_value=0.001, max_value=2.0, value=DEFAULTS["lr"], step=0.01, format="%.3f", key="train_lr")
    with st.expander("Model settings"):
        context = st.number_input("Context length", min_value=4, max_value=32, value=DEFAULTS["block_size"], step=1, help="Tokens read at once. This is also the length of the position table.")
        width = st.selectbox("Embedding width", [8, 16], index=1)
        heads = st.selectbox("Heads", [2, 4], index=1)
        blocks = st.selectbox("Blocks", [1, 2, 3], index=2)
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=DEFAULTS["seed"], step=1)
        st.caption("Width, heads, blocks, context, seed, and the corpus define the model. Changing them starts a new one. The learning rate can change between batches.")
    config = {
        "block_size": int(context),
        "n_embed": int(width),
        "num_heads": int(heads),
        "n_blocks": int(blocks),
        "seed": int(seed),
        "lr": float(lr),
    }
    try:
        planned = parameter_count(distinct, config["n_embed"], config["num_heads"], config["block_size"], config["n_blocks"])
        st.caption(f"This corpus and these sizes would allocate {planned:,} parameters.")
    except ValueError as error:
        st.error(str(error))
        return

    with st.expander("Validation file"):
        val_file = st.file_uploader("Extra text file", type=["txt"], key="val_file")
        st.caption("Optional extra file. The held-out tail of the loaded corpus is scored automatically and is not replaced by this file.")
    val_text = read_upload(val_file)[0] if val_file is not None else ""

    fp = fingerprint(corpus, source, config)
    state = model_state()
    if state is not None and state["fingerprint"] != fp:
        st.warning("The corpus or the architecture does not match the model in memory. Start training replaces that model.")

    start, stop, reset = st.columns(3)
    do_start = start.button("Start training", type="primary", width="stretch")
    do_stop = stop.button("Stop", width="stretch")
    do_reset = reset.button("Reset model", width="stretch")
    save, load, recorded = st.columns(3)
    do_save = save.button("Save checkpoint", width="stretch")
    do_load = load.button("Load checkpoint", width="stretch")
    do_recorded = recorded.button("Load recorded run", width="stretch")
    st.caption("Updates land in batches of 15 steps so the curve can refresh. Stop takes effect at the next batch. The NumPy generator state is kept with the model, including in a checkpoint.")

    if do_reset:
        st.session_state.skip_recorded = True
        for key in ("slm", "trace", "trace_meta", "generation"):
            st.session_state.pop(key, None)
        st.rerun()
    if do_load:
        try:
            st.session_state.slm = load_checkpoint()
            st.session_state.pop("trace", None)
            st.session_state.pop("generation", None)
        except (ValueError, pickle.UnpicklingError, EOFError, OSError) as error:
            st.error(str(error))
        else:
            st.rerun()
    if do_recorded:
        try:
            corpus_now, source_now = load_default_corpus()
            st.session_state.slm = load_recorded_run(ROOT / RECORDED_RUN_NAME, corpus_now, source_now)
            st.session_state.skip_recorded = False
            st.session_state.pop("trace", None)
            st.session_state.pop("generation", None)
        except (OSError, ValueError, KeyError) as error:
            st.error(str(error))
        else:
            st.rerun()
    state = model_state()
    if do_stop and state is not None:
        state["running"] = False
    if do_save:
        if state is None:
            st.error("There is no model to save.")
        else:
            path = save_checkpoint(state)
            st.caption(f"Saved {path}.")
    if do_start:
        try:
            if state is None or state["fingerprint"] != fp:
                state = initial_state(corpus, source, config)
                st.session_state.slm = state
                st.session_state.pop("trace", None)
                st.session_state.pop("generation", None)
            state["lr"] = float(lr)
            if int(steps) <= state["steps_done"]:
                state["running"] = False
                st.warning(f"The model already has {state['steps_done']} steps. Raise the step target to continue.")
            else:
                state["target"] = int(steps)
                state["running"] = True
        except ValueError as error:
            st.error(str(error))
            state = model_state()

    failed = False
    state = model_state()
    if state is not None and state.get("running") and state["steps_done"] < state["target"]:
        state["lr"] = float(lr)
        remaining = state["target"] - state["steps_done"]
        progress = st.progress(state["steps_done"] / state["target"], text="Training")

        def on_step(done, target, loss):
            progress.progress(min(done / target, 1.0), text=f"Step {done} of {target} · loss {loss:.3f}")

        try:
            train_steps(state, min(BURST, remaining), on_step)
        except Exception as error:
            failed = True
            state["running"] = False
            st.error(str(error))
        finally:
            progress.empty()
        if state["steps_done"] >= state["target"]:
            state["running"] = False

    state = model_state()
    if state is None:
        st.info("No model in memory.")
        return
    if state["steps_done"] == 0:
        st.info("A model is initialized and has not been trained.")
    elif state.get("running"):
        st.info(f"Training is in progress · step {state['steps_done']:,} of {state['target']:,}.")
    else:
        st.info(f"Trained for {state['steps_done']:,} steps on {state['source']}.")

    if state["losses"]:
        _run_metrics(state)
        st.caption(f"Elapsed {_clock(state['elapsed'])}.")
        st.plotly_chart(loss_figure(state["losses"], state["vocab_size"]), width="stretch")

    score = st.button("Score extra file")
    if val_text and score:
        try:
            remember_validation(state, val_text)
            evaluate_validation(state)
        except ValueError as error:
            st.warning(str(error))
    if state.get("file_val_loss") is not None:
        v1, v2, v3 = st.columns(3)
        v1.metric("Uploaded file loss", f"{state['file_val_loss']:.3f}")
        v2.metric("Uploaded file accuracy", f"{state['file_val_accuracy']:.1%}")
        v3.metric("Uploaded file windows", state["file_val_windows"])
        st.caption(
            f"Mean over {state['file_val_windows']} windows from the extra file, forward only. "
            f"Dropped out-of-vocabulary words: {state['val_dropped']}. This does not replace the held-out tail."
        )

    if state.get("running") and not failed:
        st.rerun()


def _clock(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {seconds - 60 * minutes:.0f}s"


def render_generate():
    page_heading("Text Generation", "The model reads the context, picks the next word from the last probability row, appends it, and repeats.")
    state = model_state()
    if state is None or state["steps_done"] == 0:
        st.info("Train a model first. A randomly initialized model is not shown here as a completion.")
        return
    examples = compatible_prompts(state["stoi"])
    if examples:
        st.caption("Examples whose words are all in the current vocabulary.")
        columns = st.columns(len(examples))
        for column, prompt in zip(columns, examples):
            if column.button(prompt, width="stretch"):
                st.session_state.gen_prompt = prompt
                st.session_state.do_generate = True
    else:
        st.caption("None of the built-in physics prompts are in this vocabulary. Type words that appear in the file.")
    if "gen_prompt" not in st.session_state:
        st.session_state.gen_prompt = examples[0] if examples else ""
    prompt = st.text_input("Prompt", key="gen_prompt")
    controls = st.columns(2)
    n_new = controls[0].slider("New tokens", 1, 12, 3)
    temperature = controls[1].slider("Temperature", 0.0, 1.5, 0.0, 0.1)
    st.caption("Temperature 0 is greedy argmax. A higher value samples from the model's probabilities in this app. The network still returns a softmax row.")
    if st.button("Generate", type="primary") or st.session_state.get("do_generate"):
        st.session_state.do_generate = False
        try:
            st.session_state.generation = generate(
                state["model"], state["stoi"], state["itos"], prompt, int(n_new), state["block_size"], float(temperature)
            )
        except ValueError as error:
            st.session_state.generation = None
            st.error(str(error))
    result = st.session_state.get("generation")
    if not generation_matches(result, prompt, temperature, n_new):
        st.session_state.generation = None
        return
    if result.get("note"):
        st.warning(result["note"])
    st.markdown(f"<div class='card'><div class='label'>Prompt</div><p>{_escape(result['prompt'])}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><div class='label'>Continuation</div><p>{_escape(result['continuation'])}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><div class='label'>Full text</div><p>{_escape(result['full'])}</p></div>", unsafe_allow_html=True)
    st.caption(
        f"Context window {state['block_size']}. {_temperature_caption(result['temperature'])} "
        "The bars are the model's raw probabilities for the first new token, before temperature."
    )
    labels = [word for word, _prob in result["top"]]
    values = [prob for _word, prob in result["top"]]
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#1d4e89"))
    figure.update_layout(
        height=240,
        margin=dict(l=8, r=8, t=28, b=8),
        title="Top five next-token probabilities",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#172033"),
        xaxis_title="Probability",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(figure, width="stretch")


def _escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_architecture():
    page_heading(
        "Architecture Explorer",
        "Decoder-only Transformer. Select a stage, then read one head's attention from an actual forward pass.",
    )
    stage_names = {item[0]: item[1] for item in STAGES}
    stage = st.selectbox("Stage", [item[0] for item in STAGES], format_func=lambda key: stage_names[key])
    pipeline([stage])
    rows = model_state()["block_size"] if model_state() else DEFAULTS["block_size"]
    st.caption(
        f"The position table has {rows} rows, one per context position. "
        f"A training script that allocates len(corpus) rows still only reads the first {rows} positions."
    )
    inner_key = None
    if stage == "block":
        st.markdown("**Inside one Transformer block**")
        inner_names = {item[0]: item[1] for item in INNER}
        inner_key = st.selectbox("Block stage", [item[0] for item in INNER], format_func=lambda key: inner_names[key])
        inner_pipeline(inner_key)
        title, class_name, role = next(item[1:] for item in INNER if item[0] == inner_key)
        equation = {
            "ln1": r"y=(x-\mu)/\sqrt{\mathrm{var}+\epsilon}\cdot\gamma+\beta",
            "attn": r"\mathrm{softmax}(\mathrm{tril}(QK^{\top}/\sqrt{d_h}))\,V",
            "res1": r"x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))",
            "ln2": r"y=(x-\mu)/\sqrt{\mathrm{var}+\epsilon}\cdot\gamma+\beta",
            "ff": r"\mathrm{ReLU}(X W_1)\,W_2",
            "res2": r"x \leftarrow x + \mathrm{FF}(\mathrm{LN}(x))",
        }[inner_key]
        _detail_card(title, class_name, role, equation, _inner_shape(inner_key))
    else:
        title, role = next((item[1], item[2]) for item in STAGES if item[0] == stage)
        class_name = {
            "tokens": "split on whitespace",
            "token_emb": "Embedding",
            "pos_emb": "PositionalEncoding",
            "sum": "Matrix add",
            "final_ln": "LayerNorm",
            "logits": "Linear",
            "probs": "Softmax",
        }[stage]
        equation = {
            "tokens": r"i_t = \mathrm{stoi}(w_t)",
            "token_emb": r"X_t = E_{i_t}",
            "pos_emb": r"P_t = E^{\mathrm{pos}}_{t}",
            "sum": r"H_t = E_{i_t} + E^{\mathrm{pos}}_{t}",
            "final_ln": r"\mathrm{LN}(H)",
            "logits": r"Z = H W_U",
            "probs": r"p_t = \mathrm{softmax}(Z_t)",
        }[stage]
        _detail_card(title, class_name, role, equation, _outer_shape(stage))

    st.subheader("Attention explorer")
    st.write(
        "Each head computes Q Kᵀ / √head_size, writes −∞ above the diagonal, and applies softmax. "
        "The heatmap is that softmax, read from the head after the forward pass. "
        "Different heads can put their mass on different keys. This view does not assign a head a semantic role; it shows the weights."
    )
    state = model_state()
    if state is None:
        if st.button("Initialize a random model on the sample corpus"):
            corpus, source = load_default_corpus()
            try:
                st.session_state.slm = initial_state(corpus, source, DEFAULTS)
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()
        st.info("Initialize or train a model to compute attention. A random model is labeled as random.")
        return
    _attention_panel(state)


def _detail_card(title, class_name, role, equation, shape_text):
    st.latex(equation)
    st.write(role)
    st.caption(f"Class: `{class_name}`. {shape_text}")


def _trace():
    return st.session_state.get("trace")


def _outer_shape(stage):
    trace = _trace()
    state = model_state()
    if trace is None:
        if state is None:
            return "Run a sequence below to fill shapes from a forward pass."
        return f"Model width {state['n_embed']}, vocabulary {state['vocab_size']}, context {state['block_size']}. Run a sequence to see this input's shapes."
    mapping = {
        "tokens": f"{len(trace['tokens'])} tokens",
        "token_emb": f"activation {_shape(trace['token_embeddings'])}, table {trace['embedding_table']}",
        "pos_emb": f"activation {_shape(trace['position_embeddings'])}",
        "sum": f"activation {_shape(trace['embedded'])}",
        "final_ln": f"activation {_shape(trace['final_ln'])}",
        "logits": f"logits {_shape(trace['logits'])}",
        "probs": f"probabilities {_shape(trace['probs'])}",
    }
    return "This forward: " + mapping.get(stage, "")


def _inner_shape(key):
    trace = _trace()
    if trace is None:
        return "Run a sequence below to fill this stage from a forward pass."
    block = trace["blocks"][0]
    mapping = {
        "ln1": block["ln1"],
        "attn": block["attention"],
        "res1": block["residual1"],
        "ln2": block["ln2"],
        "ff": block["feedforward"],
        "res2": block["output"],
    }
    return "Block 1 on the latest forward: " + _shape(mapping[key])


def _shape(rows):
    if not rows:
        return "(empty)"
    return f"({len(rows)}, {len(rows[0])})"


def _attention_panel(state):
    if state["steps_done"] == 0:
        st.warning("This model is randomly initialized. The heatmap is a real forward pass, and it is not a trained result.")
    else:
        st.caption(f"Trained model · {state['steps_done']} steps · {state['source']}.")
    default = _default_sequence(state)
    if "attn_prompt" not in st.session_state:
        st.session_state.attn_prompt = default
    prompt = st.text_input("Token sequence", key="attn_prompt")
    b_col, h_col = st.columns(2)
    block_index = b_col.selectbox("Block", list(range(state["n_blocks"])), format_func=lambda i: f"Block {i + 1}")
    head_index = h_col.selectbox("Head", list(range(state["num_heads"])), format_func=lambda i: f"Head {i + 1}")
    if st.button("Compute attention", type="primary"):
        try:
            ids, used, truncated = encode_words(state, prompt)
            trace = trace_forward(state["model"], ids, used)
            st.session_state.trace = trace
            st.session_state.trace_meta = {
                "steps": state["steps_done"],
                "prompt": " ".join(used),
                "truncated": truncated,
                "trained": state["steps_done"] > 0,
            }
        except ValueError as error:
            st.error(str(error))
    trace = _trace()
    meta = st.session_state.get("trace_meta") or {}
    if trace is None:
        st.info("Compute attention to read Q, K, V, the masked scores, and the softmax from the model.")
        return
    if meta.get("truncated"):
        st.warning(f"Only the last {state['block_size']} tokens were used.")
    if meta.get("steps") != state["steps_done"]:
        st.caption("This trace was computed at a different step. Compute again to match the current weights.")
    labels = token_labels(trace["tokens"])
    worst = 0.0
    for block in trace["blocks"]:
        for head in block["heads"]:
            worst = max(worst, causal_report(head["probs"])["max_future"])
    st.caption(f"Largest attention probability on any future token, across every head: {worst:.3e}.")
    head = trace["blocks"][block_index]["heads"][head_index]
    report = causal_report(head["probs"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sequence", len(trace["tokens"]))
    m2.metric("Head size", head["head_size"])
    m3.metric("Scale", f"{head['scale']:.4f}")
    m4.metric("Max future weight", f"{report['max_future']:.1e}")
    st.caption(
        f"Q {_shape(head['Q'])} · K {_shape(head['K'])} · V {_shape(head['V'])} · "
        f"scores {_shape(head['scores'])} · probabilities {_shape(head['probs'])}. "
        "Scale is head_size ** −0.5."
    )
    mode = st.radio("View", ["Heatmap", "Numbers"], horizontal=True)
    title = f"Block {block_index + 1} · head {head_index + 1}"
    if mode == "Heatmap":
        st.plotly_chart(
            heatmap(head["probs"], labels, labels, title, zmin=0, zmax=1),
            width="stretch",
        )
    else:
        frame = pd.DataFrame(head["probs"], index=labels, columns=labels)
        st.dataframe(frame.style.format("{:.3f}"), width="stretch")
    st.caption("Rows are queries and columns are keys. Hover a cell for the query token, the key token, and the weight. A query cannot attend to a later key.")

    st.markdown("**Heads in this block**")
    columns = st.columns(2)
    for index, other in enumerate(trace["blocks"][block_index]["heads"]):
        with columns[index % 2]:
            st.plotly_chart(
                heatmap(other["probs"], labels, labels, f"Head {index + 1}", zmin=0, zmax=1, height=300),
                width="stretch",
            )

    with st.expander("Q, K, V, scores, and the mask"):
        st.plotly_chart(feature_heatmap(head["Q"], labels, "Q"), width="stretch")
        st.caption(f"Q {_shape(head['Q'])}")
        st.plotly_chart(feature_heatmap(head["K"], labels, "K"), width="stretch")
        st.caption(f"K {_shape(head['K'])}")
        st.plotly_chart(feature_heatmap(head["V"], labels, "V"), width="stretch")
        st.caption(f"V {_shape(head['V'])}")
        scores = finite_matrix(head["scores"])
        st.plotly_chart(heatmap(scores, labels, labels, "Scaled scores before softmax", colorscale="RdBu", height=380), width="stretch")
        st.caption("Cells that were −∞ (column > row) are empty. Those positions become probability 0 in the softmax.")
        mask = [[1 if col <= row else 0 for col in range(len(labels))] for row in range(len(labels))]
        st.markdown("**Causal mask** · 1 means the score is kept")
        st.dataframe(pd.DataFrame(mask, index=labels, columns=labels), width="stretch")


def _default_sequence(state):
    for prompt in ("gravity pulls the ball", "force equals", "the ball"):
        if all(word in state["stoi"] for word in prompt.split()):
            return prompt
    return state["itos"][0]


def render_code():
    page_heading("Code & Component Playground", "Read one class at a time, then run it. Random demos say they are random. The trained model is forward-only.")
    filename = st.selectbox("File", LIBRARY_FILES)
    source = read_library(filename)
    symbols = list_symbols(source)
    labels = [f"{kind} {name}" for kind, name in symbols]
    choice = st.selectbox("Symbol", labels)
    kind, symbol = symbols[labels.index(choice)]
    extracted = extract_symbol(source, symbol)
    if extracted is None:
        st.error(f"Could not find {symbol} in {filename}.")
        return
    spec = _spec_for_file(filename, symbol)
    summary = spec["summary"] if spec else MATH_BLURBS.get(symbol, f"Defined in {filename}.")
    if filename == "dataloader.py":
        summary = "Training script. Importing this file reads poc-corpus.txt and starts a training loop, so the app never imports it. Use Train & Evaluate to train."
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.caption(f"{kind} `{symbol}` · lines {extracted[1]}–{extracted[2]} of {filename}")
        st.write(summary)
        if spec and spec.get("equation"):
            st.latex(spec["equation"])
            st.caption(f"Input {spec['input_shape']}. Output {spec['output_shape']}.")
        st.code(extracted[0], language="python")
        with st.expander(f"Full {filename} · {len(source.splitlines())} lines"):
            st.code(source, language="python")
    with right:
        st.subheader("Run")
        if filename == "dataloader.py":
            st.info("There is no runner for this file. Executing it would train immediately.")
        elif spec is None:
            st.info("This symbol is here to read. The runners cover the Transformer classes, Vector, Matrix, and the scalar helpers used by those classes.")
        else:
            _render_runner(spec)


def _spec_for_file(filename, symbol):
    for spec in SPECS:
        if spec["file"] == filename and spec["symbol"] == symbol:
            return spec
    return None


def _render_runner(spec):
    state = model_state()
    use_trained = False
    if state is not None and spec.get("trained_key"):
        use_trained = st.radio(
            "Instance",
            ["Small random demo", "Model in memory"],
            horizontal=True,
            key=f"mode-{spec['name']}",
        ) == "Model in memory"
    if use_trained:
        if state["steps_done"] == 0:
            st.warning("The model in memory has not been trained. This forward is a random initialization, not a trained result.")
        else:
            st.caption(f"Trained model after {state['steps_done']} steps. Forward only — backward would change the weights, so it is not offered here.")
        prompt = st.text_input("Prompt", value=_default_sequence(state), key=f"trained-prompt-{spec['name']}")
        block = 0
        head = 0
        if spec.get("trained_key") in BLOCK_KEYS and state["n_blocks"] > 1:
            block = st.selectbox("Block", list(range(state["n_blocks"])), format_func=lambda i: f"Block {i + 1}", key=f"block-{spec['name']}")
        if spec.get("needs_head"):
            head = st.selectbox("Head", list(range(state["num_heads"])), format_func=lambda i: f"Head {i + 1}", key=f"head-{spec['name']}")
        if st.button("Run forward", type="primary", key=f"trained-run-{spec['name']}"):
            try:
                ids, used, truncated = encode_words(state, prompt)
                trace = trace_forward(state["model"], ids, used)
                st.session_state.trace = trace
                st.session_state.trace_meta = {"steps": state["steps_done"], "prompt": " ".join(used), "truncated": truncated, "trained": state["steps_done"] > 0}
                st.session_state[f"result-{spec['name']}"] = trained_view(spec["name"], trace, block, head)
            except Exception as error:
                st.session_state[f"result-{spec['name']}"] = {"error": str(error)}
    else:
        values = _widgets(spec)
        forward = st.button("Run forward", type="primary", key=f"fwd-{spec['name']}")
        backward = False
        if spec.get("backward"):
            backward = st.button("Run backward", key=f"bwd-{spec['name']}")
        if forward or backward:
            try:
                st.session_state[f"result-{spec['name']}"] = run_component(spec["name"], values, backward=backward)
            except Exception as error:
                st.session_state[f"result-{spec['name']}"] = {"error": str(error)}
    result = st.session_state.get(f"result-{spec['name']}")
    if not result:
        st.caption("Run a pass to see shapes and values.")
        return
    if result.get("error"):
        st.error(result["error"])
        return
    if result["kind"] == "random":
        st.warning("Random initialization, seed 0, on a fresh copy. These values are not the trained model.")
    elif result["kind"] == "trained":
        if model_state() and model_state()["steps_done"] > 0:
            st.info("Trained model, forward only. Weights were not updated.")
        else:
            st.warning("Model in memory, not trained. Weights were not updated.")
    else:
        st.caption("No trainable weights in this calculation.")
    shape = result.get("shape")
    if shape not in (None, (), (0, 0)):
        st.caption(f"Shape {shape}")
    st.code(str(result["text"]), language="text")
    for label, detail in result["details"]:
        st.write(f"**{label}:** {detail}")
    if result.get("note"):
        st.caption(result["note"])


def _widgets(spec):
    values = {}
    for field in spec["fields"]:
        hide = field.get("hide_when")
        if hide and values.get(hide[0]) in hide[1]:
            continue
        key = f"play-{spec['name']}-{field['key']}"
        kind = field["kind"]
        if kind == "text":
            values[field["key"]] = st.text_input(field["label"], field["default"], key=key)
        elif kind == "area":
            values[field["key"]] = st.text_area(field["label"], field["default"], key=key, height=field.get("height", 120))
        elif kind == "number":
            values[field["key"]] = st.number_input(
                field["label"], min_value=field["min"], max_value=field["max"], value=field["default"], step=field.get("step", 1), key=key
            )
        elif kind == "choice":
            values[field["key"]] = st.selectbox(field["label"], field["choices"], key=key)
        else:
            raise ValueError(f"Unknown field {kind}.")
    return values


def _run_metrics(state):
    """Numbers read from the model in memory. Nothing here is typed in by hand."""
    if state.get("val_scored_at") != state["steps_done"] and not state.get("running"):
        score_held_out(state)
    accuracy = window_accuracy(state)
    baseline = uniform_loss(state["vocab_size"])
    a, b, c, d = st.columns(4)
    a.metric("Initial loss", f"{state['losses'][0]:.3f}")
    b.metric("Latest loss", f"{state['losses'][-1]:.3f}")
    c.metric("Validation loss (held-out tail)", "—" if state.get("val_loss") is None else f"{state['val_loss']:.3f}")
    d.metric("Uniform guess", f"{baseline:.3f}")
    e, f, g, h = st.columns(4)
    e.metric("Steps", f"{state['steps_done']:,}")
    f.metric("Learning rate", f"{state['lr']:.3f}")
    g.metric("Seed", state["seed"])
    h.metric("Window accuracy", "—" if accuracy is None else f"{accuracy:.1%}")
    if state.get("val_loss") is None:
        tail = "The last 10% of this corpus cannot fit a context window, so there is no validation loss."
    else:
        tail = (
            f"Validation loss (held-out tail) is the mean cross-entropy of {state['val_windows']} fixed windows "
            "from the last 10% of tokens, forward only. "
            f"The uniform guess is log({state['vocab_size']}) = {baseline:.2f}."
        )
    st.caption(
        "Latest loss is the last training window. The sidebar calls that value Last window loss. "
        "It is not the validation score. "
        + tail
        + " Window accuracy is the fraction of positions in the sampled training windows whose greedy next word was correct. It is not a held-out score."
    )


def _temperature_caption(temperature):
    shown = f"{float(temperature):.2f}"
    if float(temperature) <= 0:
        return f"Temperature {shown}, greedy argmax."
    return f"Sampled at temperature {shown}."


def ensure_recorded_run():
    if st.session_state.get("slm") is not None or st.session_state.get("skip_recorded"):
        return
    path = ROOT / RECORDED_RUN_NAME
    if not path.exists():
        return
    corpus, source = load_default_corpus()
    try:
        st.session_state.slm = load_recorded_run(path, corpus, source)
    except (OSError, ValueError, KeyError) as error:
        st.session_state.recorded_error = str(error)


def main():
    st.set_page_config(page_title="PhysicsSLM | Transformer From Scratch", layout="wide")
    inject_style()
    ensure_recorded_run()
    if st.session_state.get("recorded_error"):
        st.error(st.session_state.recorded_error)
    page = sidebar()
    if page == "Overview":
        render_overview()
    elif page == "Train & Evaluate":
        render_train()
    elif page == "Text Generation":
        render_generate()
    elif page == "Architecture Explorer":
        render_architecture()
    else:
        render_code()


if __name__ == "__main__":
    main()
