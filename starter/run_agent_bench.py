"""Run the agent on one split; save traces for agent_metrics.py."""
import argparse, json, time
from pathlib import Path
from agentic import run_agent
from eval_retrieval import make
from llm_controller import LLMController

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--method", required=True, choices=("tfidf", "dense", "dense_filter", "hybrid"))
p.add_argument("--split", required=True, choices=("dev", "test"))
p.add_argument("--top-k", type=int, default=3)
p.add_argument("--budget", type=int, default=800)
p.add_argument("--max-searches", type=int, default=4)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()

chunks = json.loads((ROOT / "data/converted/chunks.json").read_text(encoding="utf-8"))
cases = [c for c in json.loads((ROOT / "data/cases.json").read_text(encoding="utf-8")) if c["split"] == a.split]
retriever, controller = make(a.method, chunks), LLMController()
traces = []
for c in cases:
    t0, tok0 = time.perf_counter(), controller.tokens
    try:
        r = run_agent(c["question"], controller, retriever,
                      filters=c.get("filter") if a.method == "dense_filter" else None,
                      max_searches=a.max_searches, top_k=a.top_k, budget=a.budget)
    except Exception as e:  # record connection/model errors; never replace them with a fake answer
        r = {"status": "error", "answer": None, "citations": [], "events": [], "error": f"{type(e).__name__}: {e}"}
    traces.append({"query_id": c["id"], "question": c["question"], "model": controller.model, "method": a.method,
                   **r, "seconds": time.perf_counter() - t0, "tokens": controller.tokens - tok0})
    print(c["id"], r["status"], (r.get("answer") or r.get("error", ""))[:100])
a.output.mkdir(parents=True, exist_ok=True)
(a.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2, default=str), encoding="utf-8")