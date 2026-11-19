# Physics Tutor (CMPE 259 virtual assistant)

An LLM tutor for undergraduate mechanics and E&M. It compares four hosted or open LLMs,
with retrieval on and off, against the from-scratch PhysicsSLM, all in one Streamlit app.

## Run the app

```powershell
.venv\Scripts\python.exe -m streamlit run tutor_app.py
```

Keys are read from `.env` (local), Streamlit secrets (cloud), or environment variables:
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (or `CLAUDE_API_KEY`).
A model without a key is hidden. PhysicsSLM always works. `app.py` still opens the original
PhysicsSLM app, so the existing physicsllm.streamlit.app deployment does not change.

## Tutor lab

The Tutor lab page is the hyperparameter experiment. Four personalities (Professor Hale,
Coach Ruiz, Storyteller Cho, Examiner Shah) each have a system-prompt voice and a portrait.
They share one teaching rule: knowns, unknown, principle, symbolic equation, then the
student finishes the last step. They also share a refusal: a request for a weapon, an
explosive, or a nuclear device is turned away before retrieval or a model runs.

Open weights (Qwen 0.5B / 1.5B / 3B, Llama 1B / 3B, FP16 / 8-bit / 4-bit) and the dense
embedding models need a second install. BM25 and the hosted connectors do not.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-lab.txt
```

8-bit and 4-bit also need CUDA and `bitsandbytes`. Llama sizes need `HF_TOKEN` in `.env`
after the Hugging Face license is accepted. Qwen 0.5B is the size that runs on CPU.
Sweep one knob at a time; the page can save that comparison as CSV.

## Run the evaluation

```powershell
.venv\Scripts\python.exe run_eval.py --set mcq --limit 3     # smoke test, ~25 calls
.venv\Scripts\python.exe run_eval.py --set mcq               # 117 questions x 9 configs
.venv\Scripts\python.exe run_eval.py --set queries
.venv\Scripts\python.exe run_eval.py --set safety
```

Each reply is saved to `evaluation/raw/<set>.jsonl` as soon as it arrives, so an
interrupted run continues where it stopped. Results:

| File | Contents |
| --- | --- |
| `evaluation/summary_mcq.csv` | accuracy, 95% CI, Mechanics vs E&M, per configuration |
| `evaluation/results_mcq.csv` | every multiple-choice answer (local only, see below) |
| `evaluation/results_queries.csv` | 25 tutor queries; fill `success` (1/0), `grounded`, `failure_notes` |
| `evaluation/results_safety.csv` | 8 safety tests; fill `success` (1/0) and notes |

Re-running keeps any hand scores you typed into the two CSVs.

## Files

| File | Role |
| --- | --- |
| `rag.py` | Section-aware chunking of the OpenStax corpus + BM25 (no embedding API) |
| `llm_clients.py` | OpenAI / Gemini (incl. Gemma) / Anthropic over HTTPS with the standard library; the model table |
| `slm_adapter.py` | PhysicsSLM as a compared model: continuation, and multiple choice by likelihood |
| `tutor.py` | Prompt rules and the question → retrieval → model pipeline used by both app and eval |
| `tutor_app.py` | Streamlit app: chat, tutor lab, side-by-side compare, results, PhysicsSLM lab, about |
| `tutor_cast.py` | Four personalities, shared teaching rules, weapons refusal |
| `lab_config.py` | Hyperparameter grid for the tutor lab |
| `lab_engine.py` | One-question run: BM25 / dense / hybrid retrieval, then local or hosted generation |
| `tutor_lab.py` | Tutor lab page: portraits, two-way compare, single-knob sweep |
| `requirements-lab.txt` | Optional torch, transformers, and sentence-transformers for open weights |
| `run_eval.py` | Batch evaluation with resume, auto-scored multiple choice, CSV outputs |
| `evaluation/test_queries.csv` | 25 queries: common, needs-notes, ambiguous, edge, unreliable |
| `evaluation/safety_tests.csv` | 8 tests: injection (direct + poisoned excerpt), hallucination, scope, conflict, prompt extraction, exam cheating, false premise |
| `evaluation/mcq_workbook.csv` | 117 text-only multiple-choice items (local only) |

`mathematics.py`, `tensor.py`, `Transformer.py`, and `dataloader.py` are not modified.

## Data and licensing

- Corpus: OpenStax *University Physics* Vol. 1 and 2, CC BY 4.0.
- Multiple choice: the AP Physics practice workbook (College Board / AAPT questions). The
  workbook asks that it not be posted online, so the question CSV, per-question results, and
  the PDF are in `.gitignore`. Only the aggregate `summary_mcq.csv` is published.
- AI assistance: parts of this code and the test sets were drafted with Claude (Anthropic)
  and reviewed by the author.
