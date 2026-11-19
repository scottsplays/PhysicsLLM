"""Physics Tutor: one Streamlit app, several LLMs, RAG on or off.

Run:  streamlit run app.py

Pages
  Tutor chat          talk to one model at a time; see the retrieved sources
  Compare models      one question, every selected model side by side
  Evaluation results  multiple-choice accuracy, query and safety scoring
  PhysicsSLM lab      the from-scratch Transformer app (unchanged)
  About & design      use case, pipeline, data, models
"""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import llm_clients
import rag
import tutor
import tutor_lab

HERE = Path(__file__).resolve().parent
EVAL = HERE / "evaluation"

EXAMPLES = [
    "A 2.0 kg block is pulled across a level floor by a 10 N force against 4.0 N of friction. What is its acceleration?",
    "What does Gauss's law say, and when is it useful?",
    "A proton moves at 2.0 x 10^6 m/s perpendicular to a 0.50 T field. What is the radius of its path?",
    "How hard do I have to push the box?",
    "What is the efficiency of a Carnot engine between 500 K and 300 K?",
]


# ---------------------------------------------------------------- helpers

@st.cache_resource(show_spinner="Indexing the OpenStax corpus...")
def index():
    return rag.get_index()


def model_options():
    keys = list(llm_clients.MODELS) + list(llm_clients.EXTRA_MODELS)
    return [k for k in keys if llm_clients.has_key(llm_clients.ALL_MODELS[k]["provider"])]


def model_label(key):
    spec = llm_clients.ALL_MODELS[key]
    return f"{spec['label']}  ·  {spec['size']}"


def is_slm(key):
    return llm_clients.ALL_MODELS[key]["provider"] == "slm"


def settings_sidebar(page):
    st.sidebar.markdown("### Settings")
    if page == "chat":
        opts = model_options()
        st.sidebar.selectbox("Model", opts, key="chat_model", format_func=model_label)
    st.sidebar.toggle("Retrieval (RAG) on", value=True, key="use_rag",
                      help="On: the top-k OpenStax excerpts are added to the prompt and the "
                           "model is asked to cite them. Off: the same tutor rules, closed book.")
    st.sidebar.slider("Excerpts retrieved (k)", 1, 8, 4, key="k",
                      disabled=not st.session_state.get("use_rag", True))
    st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="temperature",
                      help="Some reasoning models only accept their default temperature; "
                           "if so the app notes that the setting was not applied.")
    missing = [p for p in ("openai", "gemini", "anthropic") if not llm_clients.has_key(p)]
    if missing:
        st.sidebar.warning("No API key for: " + ", ".join(missing) +
                           ". Those models are hidden. PhysicsSLM always works.")
    st.sidebar.divider()
    st.sidebar.caption("Course text: OpenStax *University Physics* Vol. 1 & 2, CC BY 4.0. "
                       "Mechanics + E&M only; thermodynamics, fluids, waves and optics are "
                       "deliberately out of scope.")


def show_sources(sources, expanded=False):
    if not sources:
        return
    with st.expander(f"Retrieved excerpts ({len(sources)})", expanded=expanded):
        for n, (label, score, text) in enumerate(sources, 1):
            st.markdown(f"**[S{n}] {label}**  ·  BM25 score {score:.1f}")
            st.caption(text[:1200] + ("…" if len(text) > 1200 else ""))


def meta_line(res):
    bits = []
    if res.seconds:
        bits.append(f"{res.seconds:.1f} s")
    if res.output_tokens:
        bits.append(f"{res.input_tokens:,} in / {res.output_tokens:,} out tokens")
    bits.append("RAG on" if res.use_rag and not is_slm(res.model) else
                ("RAG not applicable" if is_slm(res.model) else "RAG off"))
    if res.notes:
        bits.append(res.notes)
    return "  ·  ".join(bits)


def run_ask(model, question, history=None):
    return tutor.ask(model, question,
                     use_rag=st.session_state.get("use_rag", True),
                     k=st.session_state.get("k", 4),
                     temperature=st.session_state.get("temperature", 0.2),
                     history=history)


# ---------------------------------------------------------------- pages

def page_chat():
    index()
    settings_sidebar("chat")
    model = st.session_state.get("chat_model") or "physics-slm"
    st.title("Physics tutor")
    st.caption("Undergraduate mechanics and E&M. Pick a model in the sidebar; "
               "every model gets the same retrieved excerpts and the same tutor rules.")
    if is_slm(model):
        st.info("PhysicsSLM is the from-scratch Transformer (32-word vocabulary, 7,648 "
                "parameters). It continues the in-vocabulary words of your message; it cannot "
                "follow instructions or use retrieval. It is here as the baseline that shows "
                "what scale and instruction tuning add.")

    chat = st.session_state.setdefault("chat", [])
    cols = st.columns([1, 1, 4])
    if cols[0].button("Clear chat", disabled=not chat):
        st.session_state.chat = []
        st.rerun()

    if not chat:
        st.markdown("**Try one:**")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}"):
                st.session_state.pending = ex
                st.rerun()

    for turn in chat:
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant":
                st.caption(model_label(turn["model"]))
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                st.caption(turn["meta"])
                show_sources(turn.get("sources"))

    question = st.chat_input("Ask a mechanics or E&M question") or st.session_state.pop("pending", None)
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    history = [(chat[i]["content"], chat[i + 1]["content"])
               for i in range(0, len(chat) - 1, 2)
               if chat[i]["role"] == "user" and chat[i + 1]["role"] == "assistant"]
    with st.chat_message("assistant"):
        st.caption(model_label(model))
        with st.spinner("Thinking..."):
            res = run_ask(model, question, history)
        content = res.answer if not res.error else f"**Error:** {res.error}"
        st.markdown(content)
        st.caption(meta_line(res))
        show_sources(res.sources)
    chat.append({"role": "user", "content": question})
    chat.append({"role": "assistant", "content": content, "model": model,
                 "meta": meta_line(res), "sources": res.sources})


def page_compare():
    index()
    settings_sidebar("compare")
    st.title("Compare models")
    st.caption("One question goes to every selected model with identical retrieval, rules "
               "and temperature, so differences come from the model.")
    opts = model_options()
    default = [m for m in llm_clients.MODELS if m in opts]
    models = st.multiselect("Models", opts, default=default, format_func=model_label)
    both = st.checkbox("Also run each hosted model with RAG off (ablation)", value=False)
    question = st.text_area("Question", value=st.session_state.get("cmp_q", EXAMPLES[0]), height=90)
    if not st.button("Run comparison", type="primary", disabled=not (models and question.strip())):
        last = st.session_state.get("cmp_results")
        if last:
            render_comparison(*last)
        return
    st.session_state.cmp_q = question
    use_rag = st.session_state.get("use_rag", True)
    jobs = []
    for m in models:
        jobs.append((m, use_rag if not is_slm(m) else False))
        if both and not is_slm(m) and use_rag:
            jobs.append((m, False))
    k, temp = st.session_state.get("k", 4), st.session_state.get("temperature", 0.2)
    with st.spinner(f"Asking {len(jobs)} configurations in parallel..."):
        with ThreadPoolExecutor(len(jobs)) as pool:
            results = list(pool.map(
                lambda j: tutor.ask(j[0], question, use_rag=j[1], k=k, temperature=temp), jobs))
    st.session_state.cmp_results = (question, results)
    render_comparison(question, results)


def render_comparison(question, results):
    st.markdown(f"**Question:** {question}")
    shared = next((r.sources for r in results if r.sources), None)
    show_sources(shared)
    per_row = 3
    for start in range(0, len(results), per_row):
        cols = st.columns(per_row)
        for col, res in zip(cols, results[start:start + per_row]):
            with col.container(border=True):
                st.markdown(f"**{llm_clients.ALL_MODELS[res.model]['label']}**")
                st.caption(meta_line(res))
                if res.error:
                    st.error(res.error)
                else:
                    st.markdown(res.answer)


def load_rows(name):
    path = EVAL / name
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return pd.DataFrame(list(csv.DictReader(f)))


def page_results():
    st.title("Evaluation results")
    summary = load_rows("summary_mcq.csv")
    st.subheader("1. Multiple-choice accuracy")
    st.caption("Text-only multiple-choice questions from the AP Physics practice workbook "
               "(questions with figures removed). Each hosted model ends with 'Answer: X'; an "
               "unparseable reply counts as wrong. PhysicsSLM picks the option it finds most "
               "likely. Bars show 95% Wilson intervals; the dashed line is random guessing.")
    if summary is None or summary.empty:
        st.info("No results yet. Run `python run_eval.py --set mcq`.")
    else:
        for c in ("accuracy", "ci95_low", "ci95_high"):
            summary[c] = summary[c].astype(float)
        s = summary.sort_values("accuracy")
        fig = go.Figure(go.Bar(
            x=s["accuracy"] * 100, y=s["config"], orientation="h",
            marker_color=["#9aa5b1" if "PhysicsSLM" in c else
                          ("#1d4e89" if "RAG on" in c else "#7fa7d1") for c in s["config"]],
            error_x=dict(type="data", symmetric=False,
                         array=(s["ci95_high"] - s["accuracy"]) * 100,
                         arrayminus=(s["accuracy"] - s["ci95_low"]) * 100),
            text=[f"{a:.0%}" for a in s["accuracy"]], textposition="outside"))
        fig.add_vline(x=25, line_dash="dash", line_color="#8a4b08",
                      annotation_text="random guess (25%)", annotation_position="top")
        fig.update_layout(height=80 + 42 * len(s), margin=dict(l=10, r=40, t=30, b=40),
                          xaxis=dict(title="accuracy (%)", range=[0, 105]), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        show = summary[["config", "n", "correct", "accuracy", "ci95_low", "ci95_high",
                        "acc_Mechanics", "acc_E&M", "unparsed", "mean_seconds"]]
        st.dataframe(show, hide_index=True, width="stretch")
        detail = load_rows("results_mcq.csv")
        if detail is not None and not detail.empty:
            detail["correct"] = detail["correct"].astype(int)
            pivot = detail.pivot_table(index="config", columns="topic", values="correct",
                                       aggfunc="mean")
            st.markdown("**Accuracy by topic**")
            st.dataframe(pivot.style.format("{:.0%}").background_gradient(cmap="Blues", vmin=0, vmax=1),
                         width="stretch")

    for name, title, key in [("results_queries.csv", "2. Tutor queries (hand-scored)", "category"),
                             ("results_safety.csv", "3. Safety and robustness tests", "test_type")]:
        st.subheader(title)
        df = load_rows(name)
        if df is None or df.empty:
            st.info(f"No results yet. Run `python run_eval.py --set {'queries' if 'queries' in name else 'safety'}`.")
            continue
        scored = df[df["success"].astype(str).str.strip() != ""]
        if not scored.empty:
            scored = scored.assign(ok=scored["success"].str.strip().str.lower()
                                   .isin(["1", "y", "yes", "pass", "true"]).astype(int))
            table = scored.pivot_table(index="config", columns=key, values="ok", aggfunc="mean")
            table["overall"] = scored.groupby("config")["ok"].mean()
            st.dataframe(table.style.format("{:.0%}").background_gradient(cmap="Blues", vmin=0, vmax=1),
                         width="stretch")
        else:
            st.caption("Not scored yet: fill the `success` column (1/0) in the CSV.")
        ids = sorted(df["id"].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
        pick = st.selectbox("Inspect item", ids, key=f"pick_{name}",
                            format_func=lambda i: f"{i}: {df[df['id'] == i].iloc[0].get('query', df[df['id'] == i].iloc[0].get('prompt', ''))[:90]}")
        sub = df[df["id"] == pick]
        st.markdown(f"**Expected:** {sub.iloc[0]['expected_behavior']}")
        for _, r in sub.iterrows():
            with st.expander(f"{r['config']}  ·  success: {r.get('success') or '—'}"):
                st.markdown(r["response"] or f"_error: {r['error']}_")
                if r.get("retrieved_sections"):
                    st.caption("Retrieved: " + r["retrieved_sections"])


def page_slm_lab():
    import physics_slm_app
    original = st.set_page_config
    st.set_page_config = lambda *a, **k: None  # the app already set the page config
    try:
        physics_slm_app.main()
    finally:
        st.set_page_config = original


def page_about():
    st.title("About this tutor")
    st.markdown("""
**Use case.** A study assistant for undergraduates in calculus-based introductory physics
(mechanics and electricity & magnetism). It helps a student set up a problem (knowns,
unknown, principle, equation, result with units), explains concepts with citations to the
course textbook, asks for missing information instead of inventing it, and says when a
question is outside the course.

**Tutor lab.** Four personalities share that setup and then stop before the final number, so the student finishes the last step. The lab sweeps one hyperparameter at a time: model size, quantization, decoding, retrieval, and personality. A request for a weapon, an explosive, or a nuclear device is refused before a model is called.

**Why several models.** The project compares LLMs of very different scale and access, and
retrieval on vs. off, on the same questions with the same prompt:
""")
    rows = [{"model": v["label"], "type": v["size"], "id": v["id"]}
            for v in llm_clients.ALL_MODELS.values()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**Pipeline**")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e7eef6", color="#1d4e89", fontname="Helvetica"];
  q [label="Student question"];
  r [label="BM25 retrieval\\nOpenStax Vol. 1 & 2\\n~1,700 section-labelled chunks"];
  p [label="Prompt\\ntutor rules + [S1..Sk] excerpts\\n(or closed book)"];
  m [label="Selected LLM\\nGPT / Gemini / Claude / Gemma"];
  s [label="PhysicsSLM\\nfrom-scratch Transformer", fillcolor="#eeeeee"];
  a [label="Answer with citations"];
  q -> r -> p -> m -> a;
  q -> s [style=dashed]; s -> a [style=dashed];
}""")
    st.markdown("""
**Design choices**
- *Retrieval without an embedding API.* BM25 over chunks that never cross a textbook
  section, each labelled *Chapter › Section*. It is free, deterministic, and identical for
  every model, so the comparison isolates the model.
- *Standard-library API calls.* Each provider is one HTTPS request (`llm_clients.py`); no SDKs.
- *Retrieved text is untrusted.* The prompt tells the model excerpts are reference text, not
  instructions; a poisoned-excerpt test checks this.
- *Scope is explicit.* Thermodynamics, fluids, waves and optics are not in the corpus, which
  makes "do you admit you don't know?" testable.

**Data.** OpenStax *University Physics* Vol. 1 (ch. 1–13, 15) and Vol. 2 (ch. 5–16), CC BY 4.0,
equations linearised to plain text, figures and end-of-section problems removed. Test
questions: AP Physics practice workbook (multiple choice, text-only items), plus 25
hand-written tutor queries and 8 safety tests.

**Limitations.** Figures are not available to any model; some equations lost superscripts in
conversion; BM25 misses paraphrases that share no words with the text; hosted models may have
seen the AP questions during training.
""")


def main():
    st.set_page_config(page_title="Physics Tutor | CMPE 259", layout="wide")
    nav = st.navigation([
        st.Page(page_chat, title="Tutor chat", icon=":material/forum:", default=True),
        st.Page(tutor_lab.page, title="Tutor lab", icon=":material/tune:", url_path="tutor-lab"),
        st.Page(page_compare, title="Compare models", icon=":material/compare:"),
        st.Page(page_results, title="Evaluation results", icon=":material/bar_chart:"),
        st.Page(page_slm_lab, title="PhysicsSLM lab", icon=":material/memory:"),
        st.Page(page_about, title="About & design", icon=":material/info:"),
    ])
    nav.run()


if __name__ == "__main__":
    main()
