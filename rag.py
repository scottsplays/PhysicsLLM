"""Retrieval for the physics tutor: section-aware chunking + BM25.

No embedding API and no retrieval library. Every model in the comparison
gets exactly the same retrieved chunks for the same question, so a
difference in answers comes from the model, not from retrieval.

Corpus: OpenStax University Physics Vol. 1 and Vol. 2 (CC BY 4.0),
physics-mechanics-em-corpus.txt.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS_FILE = HERE / "physics-mechanics-em-corpus.txt"

# Mechanics chapters in the corpus; everything after these is E&M.
MECHANICS_CHAPTERS = {
    "Units and Measurement", "Vectors", "Motion Along a Straight Line",
    "Motion in Two and Three Dimensions", "Newton's Laws of Motion",
    "Applications of Newton's Laws", "Work and Kinetic Energy",
    "Potential Energy and Conservation of Energy",
    "Linear Momentum and Collisions", "Fixed-Axis Rotation",
    "Angular Momentum", "Static Equilibrium and Elasticity", "Gravitation",
    "Oscillations",
}

STOPWORDS = set("""
a an and are as at be been but by can do does for from has have how i if in
into is it its of on or so than that the their then there these this those to
was we what when where which while who why will with you your me my our they
them would could should about also just not no yes any some such very
""".split())


@dataclass
class Chunk:
    id: int
    chapter: str
    section: str
    text: str
    tokens: list = field(default_factory=list, repr=False)

    @property
    def label(self):
        return f"{self.chapter} › {self.section}"

    @property
    def subject(self):
        return "Mechanics" if self.chapter in MECHANICS_CHAPTERS else "E&M"


def tokenize(text):
    """Lowercase words and numbers, stopwords removed, crude plural strip."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", text.lower())
    out = []
    for w in words:
        w = w.replace("'s", "")
        if w in STOPWORDS or len(w) < 2 and not w.isdigit():
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _is_heading(par):
    """A heading is a one-line paragraph: short, no sentence punctuation,
    not an equation, not a worked-example label like 'Strategy: ...'."""
    if "\n" in par or len(par) > 80 or len(par) < 3:
        return False
    if par.endswith((".", ",", ";", ":", "?", "!")) or "=" in par:
        return False
    if re.match(r"^(Strategy|Solution|Significance)\b", par):
        return False
    if sum(ch.isdigit() for ch in par) > 3:
        return False
    words = par.split()
    # Sentence fragments that led into an equation removed from the corpus,
    # e.g. "Substituting these expressions into, the field becomes".
    if words[0].rstrip(",") in _LEAD_INS or words[-1].lower() in _TRAILING:
        return False
    if words[0] in ("If", "Now", "From", "Here", "As", "At", "In") and "," in par:
        return False
    return par[0].isupper()


_LEAD_INS = {"Substituting", "Solving", "Since", "So", "Taking", "Hence", "However",
             "Similarly", "Inserting", "Dividing", "Adding", "Cancelling", "Therefore",
             "Thus", "Then", "Combining", "Using", "Rearranging", "Differentiating",
             "Integrating", "Multiplying", "Writing", "Recall", "Note"}
_TRAILING = {"gives", "is", "are", "as", "becomes", "become", "yields", "obtain", "have",
             "find", "write", "get", "by", "that", "to", "the", "of", "for", "using", "us",
             "we", "be", "and", "or", "with", "where", "lasts", "velocity", "situation)"}


def chunk_corpus(text, max_chars=1200):
    """Split into chunks that never cross a section boundary.

    Paragraphs are packed until max_chars. Empty sections (the corpus has
    an empty 'Key Equations' heading at the end of every chapter, because
    the equation tables were removed) produce no chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Skip the two-line source header.
    while paragraphs and paragraphs[0].startswith(("Source:", "This file")):
        paragraphs.pop(0)

    chunks, buf = [], []
    chapter, section = "Front matter", "Introduction"

    def flush():
        if buf:
            body = "\n\n".join(buf)
            chunks.append(Chunk(len(chunks), chapter, section, body))
            buf.clear()

    for i, par in enumerate(paragraphs):
        if _is_heading(par):
            flush()
            nxt = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
            if nxt == "Introduction":  # chapter title precedes its Introduction
                chapter = par
                section = "Introduction"
            elif par != "Introduction":
                section = par
            continue
        if buf and sum(len(b) for b in buf) + len(par) > max_chars:
            flush()
        buf.append(par)
    flush()
    return chunks


class BM25:
    """Okapi BM25 over chunk text, with the section title added to each
    chunk's tokens so a question naming a topic finds that section."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        for c in chunks:
            c.tokens = tokenize(c.section + " " + c.section + " " + c.chapter + " " + c.text)
        self.tf = [Counter(c.tokens) for c in chunks]
        self.len = [len(c.tokens) for c in chunks]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out

    def search(self, query, k=4):
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(self.chunks[i], scores[i]) for i in ranked[:k] if scores[i] > 0]


_INDEX = None


def get_index(path=CORPUS_FILE):
    global _INDEX
    if _INDEX is None:
        text = Path(path).read_text(encoding="utf-8")
        _INDEX = BM25(chunk_corpus(text))
    return _INDEX


def retrieve(query, k=4):
    """Top-k (Chunk, score) pairs for a question."""
    return get_index().search(query, k)


def format_context(hits):
    """Excerpts as numbered sources [S1], [S2], ... for the prompt."""
    parts = []
    for n, (chunk, _score) in enumerate(hits, 1):
        parts.append(f"[S{n}] {chunk.label}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    idx = get_index()
    chapters = Counter(c.chapter for c in idx.chunks)
    print(f"{len(idx.chunks)} chunks across {len(chapters)} chapters")
    for q in ["How do I find the acceleration of a block with friction?",
              "What does Gauss's law say?",
              "efficiency of a Carnot engine"]:
        print("\nQ:", q)
        for c, s in retrieve(q, 3):
            print(f"  {s:6.2f}  {c.label}")
