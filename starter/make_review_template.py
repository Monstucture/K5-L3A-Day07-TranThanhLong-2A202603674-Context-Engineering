"""Write a reviews.json template next to an agent run's traces.json, for manual labelling.

Only fields prefixed with `_` are reading aids (question, expected answer, answer, cited text);
agent_metrics.py ignores them. Every `null` must be replaced by a human label before
agent_metrics.py will run. An existing reviews.json is never overwritten.
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def template(trace, case, question, chunks):
    row = {"query_id": trace["query_id"], "_status": trace["status"], "_question": trace["question"]}
    if trace["status"] == "error":
        row["_note"] = "provider error: no label needed"
        return row
    row["_answer"] = trace["answer"]
    row["_cited"] = {cid: chunks.get(cid, "(not found)") for cid in trace["citations"]}
    row["citation_support"] = [None] * len(trace["citations"])  # one true/false per citation, same order
    if case["answerable"]:
        row["_expected_answer"] = question["expected_answer"]
        row["_gold_evidence"] = [f"p{g['page']}: {g['evidence']}" for g in question["gold"]]
        # Rule from BENCHMARK.md: an abstention on an answerable question is graded 0.
        row["answer_grade"] = 0 if trace["status"] == "abstain" else None
    else:
        row["_absence_reason"] = case.get("absence_reason")
        row["abstention_correct"] = None
    row["reason"] = ""
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path, help="agent run folders containing traces.json")
    a = p.parse_args()
    cases = {c["id"]: c for c in json.loads((ROOT / "data/cases.json").read_text(encoding="utf-8"))}
    questions = {q["id"]: q for q in json.loads((ROOT / "data/converted/questions.json").read_text(encoding="utf-8"))}
    chunks = {c["id"]: c["text"] for c in json.loads((ROOT / "data/converted/chunks.json").read_text(encoding="utf-8"))}
    for run in a.runs:
        out = run / "reviews.json"
        if out.exists():
            print(f"skip {out}: already exists")
            continue
        traces = json.loads((run / "traces.json").read_text(encoding="utf-8"))
        rows = [template(t, cases[t["query_id"]], questions.get(t["query_id"]), chunks) for t in traces]
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        todo = sum(json.dumps(r).count("null") for r in rows)
        print(f"wrote {out}: {len(rows)} questions, {todo} labels to fill")


if __name__ == "__main__":
    main()
