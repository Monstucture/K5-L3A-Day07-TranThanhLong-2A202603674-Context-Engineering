import json
from pathlib import Path

# Create a single cases.json file from the converted questions and unanswerable questions.

D = Path(__file__).resolve().parents[1] / "data"
qs = json.loads((D / "converted/questions.json").read_text(encoding="utf-8"))
un = json.loads((D / "unanswerable.json").read_text(encoding="utf-8"))
cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
          "filter": q.get("filter")} for q in qs] + [{**u, "answerable": False} for u in un]
(D / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(cases), "cases")