"""One lab question: retrieve with the selected index, then generate.

Open-weight models and embedding models are imported only when a run asks
for them, so the rest of the app still starts with the base requirements.
BM25 reuses the section-aware chunker in rag.py.
"""

import gc
import os
from dataclasses import dataclass, field

import llm_clients
import rag
import tutor_cast
from lab_config import EMBEDDINGS, MODEL_SIZES

POOL = 20
_CHUNKS = {}
_BM25 = {}
_DENSE = {}
_RERANKER = None
_LOCAL = None  # (key, model, tokenizer) — one open-weight model at a time


@dataclass
class LabResult:
    summary: str
    question: str
    answer: str
    prompt: str = ""
    sources: list = field(default_factory=list)  # [(label, score, text)]
    seconds: float = 0.0
    error: str = ""
    notes: str = ""


def chunks_for(chunk_size):
    """Section-bounded chunks packed to about chunk_size characters."""
    if chunk_size not in _CHUNKS:
        text = rag.CORPUS_FILE.read_text(encoding="utf-8")
        _CHUNKS[chunk_size] = rag.chunk_corpus(text, max_chars=chunk_size)
    return _CHUNKS[chunk_size]


def _bm25(chunk_size):
    if chunk_size not in _BM25:
        _BM25[chunk_size] = rag.BM25(chunks_for(chunk_size))
    return _BM25[chunk_size]


def rrf(rank_lists, k=60):
    """Reciprocal rank fusion. Each list is best-first chunk indexes."""
    scores = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])


def _dense(embedding, chunk_size):
    key = (embedding, chunk_size)
    if key not in _DENSE:
        # Drop the previous matrix before loading another embedding model.
        _DENSE.clear()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDINGS[embedding])
        chunks = chunks_for(chunk_size)
        matrix = model.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False,
        )
        _DENSE[key] = (model, matrix)
    return _DENSE[key]


def _dense_order(query, embedding, chunk_size, n):
    model, matrix = _dense(embedding, chunk_size)
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = matrix @ q
    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    return order[:n], scores


def _rerank(query, pairs, top_k):
    """pairs are (chunk, retrieval_score). Returns the same shape, top_k long."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    if not pairs:
        return []
    raw = _RERANKER.predict([(query, c.text) for c, _ in pairs], show_progress_bar=False)
    ranked = sorted(zip(pairs, raw), key=lambda item: -float(item[1]))
    return [(chunk, float(score)) for (chunk, _), score in ranked[:top_k]]


def retrieve(query, config):
    """Top-k (Chunk, score) pairs for this retrieval configuration."""
    chunks = chunks_for(config.chunk_size)
    n = POOL if config.rerank else config.top_k
    strategy = config.strategy

    if strategy == "BM25":
        index = _bm25(config.chunk_size)
        scores = index.scores(query)
        order = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i]) if scores[i] > 0]
        picked = [(chunks[i], float(scores[i])) for i in order[:n]]
    elif strategy == "Dense":
        order, scores = _dense_order(query, config.embedding, config.chunk_size, n)
        picked = [(chunks[i], float(scores[i])) for i in order]
    elif strategy == "Hybrid":
        index = _bm25(config.chunk_size)
        lexical = index.scores(query)
        lex_order = sorted(range(len(lexical)), key=lambda i: -lexical[i])[:POOL]
        den_order, dense_scores = _dense_order(query, config.embedding, config.chunk_size, POOL)
        fused = rrf([lex_order, den_order])[:n]
        picked = []
        for i in fused:
            # Keep a readable score: dense cosine when the chunk was in that pool.
            score = float(dense_scores[i]) if i in den_order else float(lexical[i])
            picked.append((chunks[i], score))
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    if config.rerank:
        picked = _rerank(query, picked, config.top_k)
    return picked[:config.top_k]


def _user_message(question, hits):
    parts = []
    if hits:
        parts.append(rag.format_context(hits))
    else:
        parts.append("(No excerpts retrieved.)")
    parts.append("Student question:\n" + question)
    return "\n\n=====\n\n".join(parts)


def _hf_token():
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    dot = llm_clients._read_dotenv()
    return dot.get("HF_TOKEN") or dot.get("HUGGING_FACE_HUB_TOKEN") or None


def _load_local(repo, quant):
    """Load one instruct model. FP16 is FP32 on a machine without CUDA."""
    global _LOCAL
    key = (repo, quant)
    if _LOCAL and _LOCAL[0] == key:
        return _LOCAL[1], _LOCAL[2], ""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _LOCAL:
        _LOCAL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cuda = torch.cuda.is_available()
    note = ""
    kwargs = {"token": _hf_token()}
    if quant == "FP16":
        if cuda:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
            note = "FP16 on a CPU loads as FP32. 8-bit and 4-bit need a CUDA GPU."
    else:
        if not cuda:
            raise RuntimeError(
                "8-bit and 4-bit quantization need a CUDA GPU (and bitsandbytes). "
                "On this machine, use FP16."
            )
        from transformers import BitsAndBytesConfig
        if quant == "8-bit":
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "4-bit":
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            raise ValueError(f"Unknown quantization: {quant}")
        kwargs["quantization_config"] = qcfg
        kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(repo, token=kwargs.get("token"))
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    model.eval()
    _LOCAL = (key, model, tokenizer)
    return model, tokenizer, note


def _generate_local(config, system, user):
    try:
        import torch
    except ImportError:
        return "", 0.0, (
            "Open-weight models need the lab install: "
            r".venv\Scripts\python.exe -m pip install -r requirements-lab.txt"
        ), ""

    repo = MODEL_SIZES[config.model_size]
    try:
        model, tokenizer, note = _load_local(repo, config.quantization)
    except Exception as e:  # noqa: BLE001 — surface download and license errors
        text = str(e)
        if "gated" in text.lower() or "401" in text or "403" in text:
            text = (
                "Those Llama weights are gated. Accept the license on the model page "
                "and set HF_TOKEN in .env. The Qwen sizes download without a token. "
                f"Original error: {text[:300]}"
            )
        return "", 0.0, text, ""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    limit = 3072 if torch.cuda.is_available() else 1536
    if input_ids.shape[1] > limit:
        input_ids = input_ids[:, -limit:]
        note = (note + " " if note else "") + f"Prompt trimmed to the last {limit} tokens."

    sampling = config.temperature > 0
    gen = {
        "max_new_tokens": int(config.max_tokens),
        "do_sample": sampling,
        "repetition_penalty": float(config.repetition_penalty),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sampling:
        gen["temperature"] = float(config.temperature)
        gen["top_p"] = float(config.top_p)
    else:
        note = (note + " " if note else "") + "Temperature 0 uses greedy decoding, so top-p is not applied."

    start = __import__("time").time()
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen)
    elapsed = __import__("time").time() - start
    new_ids = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, elapsed, "", note


def _generate_hosted(config, system, user):
    reply = llm_clients.chat(
        config.hosted_model, system, user,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
    )
    notes = []
    if config.repetition_penalty != 1.0:
        notes.append("Repetition penalty is an open-weight setting and was not sent.")
    if config.quantization != "FP16":
        notes.append("Quantization is an open-weight setting and was not sent.")
    if reply.dropped_params:
        notes.append("API ignored: " + ", ".join(reply.dropped_params))
    return reply.text, reply.seconds, reply.error, " ".join(notes)


def run(config, question):
    """Answer one question. A weapons request never reaches a model."""
    summary = config.summary()
    if tutor_cast.blocks_weapons(question):
        return LabResult(
            summary, question, tutor_cast.REFUSAL,
            notes="Refused before any model or retrieval call.",
        )

    try:
        hits = retrieve(question, config)
    except ImportError:
        return LabResult(
            summary, question, "",
            error="Dense, Hybrid, and reranking need the lab install "
                  r"(.venv\Scripts\python.exe -m pip install -r requirements-lab.txt). "
                  "BM25 works with the base install.",
        )
    except Exception as e:  # noqa: BLE001 — download and runtime errors
        return LabResult(summary, question, "", error=f"Retrieval failed: {e}")

    sources = [(c.label, s, c.text) for c, s in hits]
    system = tutor_cast.system_prompt(config.personality, has_notes=bool(hits))
    user = _user_message(question, hits)
    try:
        if config.backend == "local":
            text, seconds, error, notes = _generate_local(config, system, user)
        elif config.backend == "hosted":
            text, seconds, error, notes = _generate_hosted(config, system, user)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
    except Exception as e:  # noqa: BLE001
        return LabResult(summary, question, "", system, sources, error=str(e))
    return LabResult(summary, question, text, system, sources, seconds, error, notes)
