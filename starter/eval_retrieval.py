"""Retrieval-only benchmark: tfidf | dense | dense_filter | hybrid | dense_rerank."""
import argparse, json, statistics, time
from pathlib import Path
from baseline import TfidfIndex, hit, pack_context

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("tfidf", "dense", "dense_filter", "hybrid", "dense_rerank")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small English cross-encoder, runs on CPU


class Tfidf:
    def __init__(self, chunks):
        self.index = TfidfIndex(chunks)

    def search(self, query, *, filters=None, top_k=3):
        return self.index.search(query, top_k=top_k)


def rrf(*rankings, k=60, weights=None):
    """Weighted Reciprocal Rank Fusion: each list adds weight / (k + rank); score scales don't matter."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores, by_id = {}, {}
    for weight, ranking in zip(weights, rankings):
        for rank, c in enumerate(ranking):
            scores[c["id"]] = scores.get(c["id"], 0) + weight / (k + rank + 1)
            by_id.setdefault(c["id"], c)
    return [{**by_id[i], "score": s} for i, s in sorted(scores.items(), key=lambda x: -x[1])]


class Hybrid:
    def __init__(self, dense, tfidf, pool=20, lexical_weight=1.0):
        self.dense, self.tfidf, self.pool, self.lexical_weight = dense, tfidf, pool, lexical_weight

    def search(self, query, *, filters=None, top_k=3):
        return rrf(self.dense.search(query, filters=filters, top_k=self.pool),
                   self.tfidf.search(query, top_k=self.pool),
                   weights=[1.0, self.lexical_weight])[:top_k]


class Rerank:
    """Take `pool` candidates from a base retriever, re-score each (query, chunk) pair with a cross-encoder."""

    def __init__(self, base, model_name=RERANK_MODEL, pool=20):
        try:
            from sentence_transformers import CrossEncoder  # only this method needs it
        except ImportError as error:
            raise RuntimeError("dense_rerank needs `pip install sentence-transformers`") from error
        self.base, self.pool, self.model_name = base, pool, model_name
        self.model = CrossEncoder(model_name)

    def search(self, query, *, filters=None, top_k=3):
        candidates = self.base.search(query, filters=filters, top_k=self.pool)
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates])
        # Keep the first-stage rank/score in the trace so before/after can be compared.
        ranked = [{**c, "retrieval_rank": i + 1, "retrieval_score": c.get("score"), "score": float(s)}
                  for i, (c, s) in enumerate(zip(candidates, scores))]
        return sorted(ranked, key=lambda c: -c["score"])[:top_k]


def make(method, chunks, lexical_weight=1.0, rerank_model=RERANK_MODEL, pool=20):
    if method == "tfidf":
        return Tfidf(chunks)
    from day6_retriever import Day6Retriever  # imported here so tests and TF-IDF runs need no Day 6 setup
    dense = Day6Retriever()
    if method == "hybrid":
        return Hybrid(dense, Tfidf(chunks), pool=pool, lexical_weight=lexical_weight)
    if method == "dense_rerank":
        return Rerank(dense, rerank_model, pool=pool)
    return dense


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--budget", type=int, default=800)
    p.add_argument("--pool", type=int, default=20, help="candidates per retriever for hybrid / dense_rerank")
    p.add_argument("--lexical-weight", type=float, default=1.0, help="hybrid: TF-IDF weight in RRF (dense = 1.0)")
    p.add_argument("--rerank-model", default=RERANK_MODEL)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    chunks = json.loads((ROOT / "data/converted/chunks.json").read_text(encoding="utf-8"))
    questions = [q for q in json.loads((ROOT / "data/converted/questions.json").read_text(encoding="utf-8"))
                 if q["split"] == a.split]
    retriever = make(a.method, chunks, a.lexical_weight, a.rerank_model, a.pool)
    traces = []
    for q in questions:
        filters = q.get("filter") if a.method == "dense_filter" else None  # filter given with the question, not from gold
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=a.top_k)
        ms = (time.perf_counter() - t0) * 1000
        context, ids = pack_context(results, a.budget)
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        traces.append({"query_id": q["id"], "question": q["question"], "method": a.method, "filter": filters,
                       "results": marked, "context_ids": ids, "context_chars": len(context), "ms": ms,
                       "hit1": bool(marked) and marked[0]["hit"],
                       "hit3": any(c["hit"] for c in marked[:3]),
                       "context_hit": any(c["hit"] and c["id"] in ids for c in marked)})
    n = len(traces)
    summary = {"method": a.method, "split": a.split, "top_k": a.top_k, "budget": a.budget, "n": n,
               **({"pool": a.pool, "lexical_weight": a.lexical_weight} if a.method == "hybrid" else {}),
               **({"pool": a.pool, "rerank_model": a.rerank_model} if a.method == "dense_rerank" else {}),
               **{m: f"{sum(t[m] for t in traces)}/{n}" for m in ("hit1", "hit3", "context_hit")},
               "median_ms": statistics.median(t["ms"] for t in traces)}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()