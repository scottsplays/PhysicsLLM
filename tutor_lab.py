"""Tutor lab page: four personalities, and one knob at a time.

Open-weight runs follow the Qwen notebook (local model, local embeddings).
Hosted runs use the OpenAI and Gemini connectors already in llm_clients.py.
"""

import csv
import io

import streamlit as st

import llm_clients
import tutor_cast
from lab_config import (
    CHUNK_SIZES, DOWNLOAD_AXES, EMBEDDINGS, LOCAL_ONLY_AXES, MAX_TOKENS,
    MODEL_SIZES, QUANTIZATION, REPETITION_PENALTIES, STRATEGIES, SWEEP_AXES,
    TEMPERATURES, TOP_K, TOP_P, LabConfig,
)
from lab_engine import run

_ORDER = ["professor", "coach", "storyteller", "examiner"]


def _pick(label, options, default, key, fmt=str):
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, key=key, format_func=fmt)


def _form(prefix, personality):
    """The twelve-axis grid, as discrete choices, for one side of a comparison."""
    st.markdown("**Model**")
    backend = st.radio(
        "Backend", ["Open weights", "Hosted"],
        horizontal=True, key=f"{prefix}_backend",
        help="Open weights are the Qwen and Llama sizes. Hosted models are the "
             "OpenAI and Gemini connectors from the other two notebooks.",
    )
    local = backend == "Open weights"
    if local:
        model_size = _pick("Model size", list(MODEL_SIZES), "Qwen 0.5B", f"{prefix}_size")
        quant = _pick("Quantization", QUANTIZATION, "FP16", f"{prefix}_quant")
        hosted = "gemini-3.8-flash"
        st.caption("Qwen 0.5B is the size that fits a CPU. The first run downloads the weights. "
                   "8-bit and 4-bit need a CUDA GPU. Llama 1B and 3B need HF_TOKEN after you "
                   "accept the license on Hugging Face.")
    else:
        hosted_keys = [k for k in list(llm_clients.MODELS) + list(llm_clients.EXTRA_MODELS)
                       if llm_clients.ALL_MODELS[k]["provider"] != "slm"]
        hosted = _pick(
            "Hosted model", hosted_keys, "gemini-3.8-flash", f"{prefix}_hosted",
            fmt=lambda k: llm_clients.ALL_MODELS[k]["label"],
        )
        model_size, quant = "Qwen 0.5B", "FP16"
        st.caption("Quantization and repetition penalty stay on the open-weight side. "
                   "Hosted calls send temperature, top-p, and max tokens.")

    st.markdown("**Generation**")
    g1, g2 = st.columns(2)
    with g1:
        temperature = _pick("Temperature", TEMPERATURES, 0.3, f"{prefix}_temp")
        max_tokens = _pick("Max output tokens", MAX_TOKENS, 256, f"{prefix}_max")
    with g2:
        top_p = _pick("Top-p", TOP_P, 0.9, f"{prefix}_topp")
        if local:
            penalty = _pick("Repetition penalty", REPETITION_PENALTIES, 1.0, f"{prefix}_rep")
        else:
            penalty = 1.0
            st.caption("Repetition penalty applies when Backend is Open weights.")

    st.markdown("**Retrieval**")
    r1, r2 = st.columns(2)
    with r1:
        embedding = _pick("Embedding model", list(EMBEDDINGS), "MiniLM", f"{prefix}_emb")
        strategy = _pick("Retrieval strategy", STRATEGIES, "BM25", f"{prefix}_strat")
        rerank = st.selectbox("Reranking", ["Off", "On"], key=f"{prefix}_rerank") == "On"
    with r2:
        top_k = _pick("Top-k retrieval", TOP_K, 5, f"{prefix}_k")
        chunk_size = _pick("Chunk size", CHUNK_SIZES, 1000, f"{prefix}_chunk")
    if strategy == "BM25":
        st.caption("BM25 does not use the embedding model. Switch to Dense or Hybrid to compare MiniLM, BGE-small, and MPNet.")
    else:
        st.caption("Dense and Hybrid download the embedding model on first use. Hybrid fuses it with BM25.")

    if prefix == "b":
        st.markdown("**Teaching**")
        personality = _pick(
            "Tutor personality", _ORDER, personality, f"{prefix}_who",
            fmt=lambda k: tutor_cast.CAST[k].name,
        )

    return LabConfig(
        personality=personality,
        backend="local" if local else "hosted",
        model_size=model_size,
        hosted_model=hosted,
        quantization=quant,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        repetition_penalty=float(penalty),
        embedding=embedding,
        strategy=strategy,
        top_k=int(top_k),
        chunk_size=int(chunk_size),
        rerank=rerank,
    )


def _cast_picker():
    st.markdown("**Teaching — pick a tutor.** The portrait is the personality. "
                "The system prompt carries that voice, and every voice keeps the same teaching goal.")
    cols = st.columns(4)
    chosen = st.session_state.setdefault("lab_personality", "professor")
    for col, key in zip(cols, _ORDER):
        person = tutor_cast.CAST[key]
        with col:
            st.image(str(person.portrait), width=180)
            st.markdown(f"**{person.name}**")
            st.caption(person.tagline)
            if st.button("Use this tutor" if key != chosen else "Selected",
                         key=f"cast_{key}", type="primary" if key == chosen else "secondary"):
                st.session_state.lab_personality = key
                st.rerun()
    return st.session_state.lab_personality


def _show(result):
    st.caption(result.summary)
    if result.error:
        st.error(result.error)
    if result.answer:
        st.markdown(result.answer)
    bits = []
    if result.seconds:
        bits.append(f"{result.seconds:.1f} s")
    if result.notes:
        bits.append(result.notes)
    if bits:
        st.caption(" · ".join(bits))
    if result.sources:
        with st.expander(f"Retrieved excerpts ({len(result.sources)})"):
            for n, (label, score, text) in enumerate(result.sources, 1):
                st.markdown(f"**[S{n}] {label}**  ·  {score:.3f}")
                st.caption(text[:900] + ("…" if len(text) > 900 else ""))
    if result.prompt:
        with st.expander("System prompt"):
            st.text(result.prompt)


def _csv(results):
    buf = io.StringIO()
    fields = ["summary", "question", "answer", "seconds", "error", "notes", "sources"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow({
            "summary": r.summary,
            "question": r.question,
            "answer": r.answer,
            "seconds": f"{r.seconds:.2f}",
            "error": r.error,
            "notes": r.notes,
            "sources": " | ".join(label for label, _, _ in r.sources),
        })
    return buf.getvalue()


def page():
    st.title("Tutor lab")
    st.caption(
        "Same course, four personalities, and the hyperparameter grid from the three "
        "notebooks. Ask once, compare two settings, or sweep a single knob. "
        "Every tutor stops before the final number. A request for a weapon, an explosive, "
        "or a nuclear device is refused before a model is called."
    )
    personality = _cast_picker()
    person = tutor_cast.CAST[personality]
    st.info(f"{person.name} is selected. {person.tagline}")

    compare = st.toggle("Compare a second configuration", value=False,
                        help="The right side starts as Coach Ruiz with the same knobs, so the first difference you see is personality.")
    left, right = st.columns(2) if compare else (st.container(), None)
    with left:
        if compare:
            st.subheader(person.name)
        config_a = _form("a", personality)
    config_b = None
    if compare:
        with right:
            other = "coach" if personality != "coach" else "professor"
            st.subheader("Second configuration")
            config_b = _form("b", other)

    question = st.text_area(
        "Question",
        value="A 2.0 kg block is pulled across a level floor by a 10 N force against 4.0 N of friction. What is its acceleration?",
        height=90, key="lab_question",
    )

    axes = [name for name in SWEEP_AXES if config_a.backend == "local" or name not in LOCAL_ONLY_AXES]
    sweep_name = st.selectbox("Knob to sweep", axes, key="lab_axis")
    if sweep_name in DOWNLOAD_AXES:
        st.caption("This sweep can download a model the first time each value runs. "
                   "Qwen 0.5B and BM25 are the settings that stay small.")

    go, sweep, _ = st.columns([1, 1, 2])
    ask = go.button("Ask", type="primary", disabled=not question.strip())
    do_sweep = sweep.button("Sweep this knob", disabled=not question.strip())

    if ask:
        configs = [config_a] + ([config_b] if config_b else [])
        results = []
        for cfg in configs:
            with st.spinner(cfg.summary()):
                results.append(run(cfg, question.strip()))
        st.session_state.lab_results = results

    if do_sweep:
        field, values = SWEEP_AXES[sweep_name]
        results = []
        for value in values:
            cfg = config_a.vary(field, value)
            with st.spinner(cfg.summary()):
                results.append(run(cfg, question.strip()))
        st.session_state.lab_results = results

    results = st.session_state.get("lab_results")
    if not results:
        return
    st.divider()
    columns = st.columns(len(results))
    for col, result in zip(columns, results):
        with col:
            _show(result)
    st.download_button(
        "Download this comparison (CSV)",
        data=_csv(results),
        file_name="tutor_lab_comparison.csv",
        mime="text/csv",
    )
