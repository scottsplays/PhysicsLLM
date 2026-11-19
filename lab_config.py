"""The knobs on the Tutor lab page.

Values are the comparison grid, not a full factorial. The page runs one
configuration, a pair, or a sweep of a single knob.
"""

from dataclasses import dataclass, replace

import tutor_cast

# Open weights used by the Qwen notebook, plus the Llama sizes in the grid.
# Qwen downloads without an account. Llama 3.2 instruct weights are gated.
MODEL_SIZES = {
    "Qwen 0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen 1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen 3B": "Qwen/Qwen2.5-3B-Instruct",
    "Llama 1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama 3B": "meta-llama/Llama-3.2-3B-Instruct",
}

QUANTIZATION = ["FP16", "8-bit", "4-bit"]
TEMPERATURES = [0.0, 0.3, 0.7]
TOP_P = [0.8, 0.9, 1.0]
MAX_TOKENS = [128, 256, 512]
REPETITION_PENALTIES = [1.0, 1.05, 1.1]

EMBEDDINGS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE-small": "BAAI/bge-small-en-v1.5",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
}
STRATEGIES = ["Dense", "BM25", "Hybrid"]
TOP_K = [3, 5, 10]
CHUNK_SIZES = [500, 1000, 1500]

# One axis at a time. Model size and quantization are local-only; the page
# hides them when the backend is a hosted notebook model.
SWEEP_AXES = {
    "Model size": ("model_size", list(MODEL_SIZES)),
    "Quantization": ("quantization", list(QUANTIZATION)),
    "Temperature": ("temperature", list(TEMPERATURES)),
    "Top-p": ("top_p", list(TOP_P)),
    "Max output tokens": ("max_tokens", list(MAX_TOKENS)),
    "Repetition penalty": ("repetition_penalty", list(REPETITION_PENALTIES)),
    "Embedding model": ("embedding", list(EMBEDDINGS)),
    "Retrieval strategy": ("strategy", list(STRATEGIES)),
    "Top-k retrieval": ("top_k", list(TOP_K)),
    "Chunk size": ("chunk_size", list(CHUNK_SIZES)),
    "Reranking": ("rerank", [False, True]),
    "Tutor personality": ("personality", list(tutor_cast.CAST)),
}

LOCAL_ONLY_AXES = {"Model size", "Quantization", "Repetition penalty"}
DOWNLOAD_AXES = {"Model size", "Quantization", "Embedding model", "Retrieval strategy", "Reranking"}


@dataclass
class LabConfig:
    personality: str = "professor"
    backend: str = "local"          # "local" or "hosted"
    model_size: str = "Qwen 0.5B"
    hosted_model: str = "gemini-3.8-flash"
    quantization: str = "FP16"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 256
    repetition_penalty: float = 1.0
    embedding: str = "MiniLM"
    strategy: str = "BM25"
    top_k: int = 5
    chunk_size: int = 1000
    rerank: bool = False

    def vary(self, field, value):
        return replace(self, **{field: value})

    def summary(self):
        who = tutor_cast.CAST[self.personality].name
        if self.backend == "local":
            model = f"{self.model_size}, {self.quantization}"
        else:
            model = self.hosted_model
        return (
            f"{who} · {model} · temperature {self.temperature:g} · top-p {self.top_p:g} · "
            f"max {self.max_tokens} · repetition {self.repetition_penalty:g} · "
            f"{self.embedding} · {self.strategy} · k={self.top_k} · "
            f"chunk {self.chunk_size} · rerank {'on' if self.rerank else 'off'}"
        )
