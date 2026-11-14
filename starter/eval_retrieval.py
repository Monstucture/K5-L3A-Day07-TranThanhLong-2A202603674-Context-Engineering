"""Retrieval-only benchmark: tfidf | dense | dense_filter | hybrid."""
import argparse, json, statistics, time
from pathlib import Path
from baseline import TfidfIndex, hit, pack_context
from day6_retriever import Day6Retriever

ROOT = Path(__file__).resolve().parents[1]


class Tfidf:
    def __init__(self, chunks):
        self.index = TfidfIndex(chunks)

    def search(self, query, *, filters=None, top_k=3):
        return self.index.search(query, top_k=top_k)


def rrf(*rankings, k=60):
    """Reciprocal Rank Fusion: combines rankings by rank, so different score scales don't matter."""
    scores, by_id = {}, {}
    for ranking in rankings:
        for rank, c in enumerate(ranking):
            scores[c["id"]] = scores.get(c["id"], 0) + 1 / (k + rank + 1)
            by_id.setdefault(c["id"], c)
    return [{**by_id[i], "score": s} for i, s in sorted(scores.items(), key=lambda x: -x[1])]


class Hybrid:
    def __init__(self, dense, tfidf, pool=20):
        self.dense, self.tfidf, self.pool = dense, tfidf, pool

    def search(self, query, *, filters=None, top_k=3):
        return rrf(self.dense.search(query, filters=filters, top_k=self.pool),
                   self.tfidf.search(query, top_k=self.pool))[:top_k]


def make(method, chunks):
    if method == "tfidf":
        return Tfidf(chunks)
    dense = Day6Retriever()
    return Hybrid(dense, Tfidf(chunks)) if method == "hybrid" else dense


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=("tfidf", "dense", "dense_filter", "hybrid"))
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--budget", type=int, default=800)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    chunks = json.loads((ROOT / "data/converted/chunks.json").read_text(encoding="utf-8"))
    questions = [q for q in json.loads((ROOT / "data/converted/questions.json").read_text(encoding="utf-8"))
                 if q["split"] == a.split]
    retriever = make(a.method, chunks)
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
               **{m: f"{sum(t[m] for t in traces)}/{n}" for m in ("hit1", "hit3", "context_hit")},
               "median_ms": statistics.median(t["ms"] for t in traces)}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()