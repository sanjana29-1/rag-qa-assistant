"""A small Retrieval-Augmented Generation (RAG) question-answering assistant.

Pipeline:  load docs -> chunk by section -> embed (TF-IDF) -> retrieve top-k by
cosine similarity -> generate an answer grounded in the retrieved chunks.

Generation uses the Anthropic API if ANTHROPIC_API_KEY is set; otherwise it
falls back to an extractive answer (the most relevant sentences), so the
project runs fully offline.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    title: str
    text: str


def load_chunks(path: str | Path) -> list[Chunk]:
    """Split a markdown file into one chunk per '## ' section."""
    chunks, title, lines = [], None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if title:
                chunks.append(Chunk(title, " ".join(lines).strip()))
            title, lines = line[3:].strip(), []
        elif title and line.strip():
            lines.append(line.strip())
    if title:
        chunks.append(Chunk(title, " ".join(lines).strip()))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        # Title is included so section names help matching
        self.matrix = self.vectorizer.fit_transform(f"{c.title}. {c.title}. {c.text}" for c in chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def extractive_answer(query: str, hits: list[tuple[Chunk, float]], retriever: Retriever) -> str:
    """Offline fallback: return the sentence(s) most similar to the question."""
    sentences = [(s, c.title) for c, _ in hits for s in re.split(r"(?<=[.!?])\s+", c.text) if s]
    vecs = retriever.vectorizer.transform([s for s, _ in sentences])
    scores = cosine_similarity(retriever.vectorizer.transform([query]), vecs)[0]
    order = np.argsort(scores)[::-1][:2]
    # keep the runner-up sentence only if it is nearly as relevant as the best one
    best = [sentences[i] for i in order if scores[i] > 0 and scores[i] >= 0.8 * scores[order[0]]]
    return " ".join(s for s, _ in best) if best else "I couldn't find that in the knowledge base."


def llm_answer(query: str, hits: list[tuple[Chunk, float]]) -> str:
    import anthropic  # optional dependency

    context = "\n\n".join(f"[{c.title}]\n{c.text}" for c, _ in hits)
    msg = anthropic.Anthropic().messages.create(
        model=os.getenv("RAG_MODEL", "claude-haiku-4-5"),
        max_tokens=300,
        system="Answer ONLY from the provided context. If the answer is not in it, say you don't know. "
               "Cite the section name in square brackets.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return msg.content[0].text


class RAGAssistant:
    def __init__(self, kb_path: str | Path = "knowledge_base.md", k: int = 3, min_score: float = 0.05):
        self.retriever = Retriever(load_chunks(kb_path))
        self.k, self.min_score = k, min_score

    def ask(self, query: str) -> dict:
        hits = [h for h in self.retriever.search(query, self.k) if h[1] >= self.min_score]
        if not hits:
            return {"answer": "I couldn't find that in the knowledge base.", "sources": []}
        use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
        answer = llm_answer(query, hits) if use_llm else extractive_answer(query, hits, self.retriever)
        return {"answer": answer, "sources": [(c.title, round(s, 3)) for c, s in hits]}


if __name__ == "__main__":
    bot = RAGAssistant()
    questions = sys.argv[1:] or ["Can I get a cash refund on a sale item?"]
    for q in questions:
        r = bot.ask(q)
        print(f"Q: {q}\nA: {r['answer']}\nSources: {r['sources']}\n")
