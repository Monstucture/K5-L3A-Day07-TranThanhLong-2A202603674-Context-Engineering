import json
from pathlib import Path
from day6_retriever import Day6Retriever

store = Day6Retriever().store
with store._conn.cursor() as cur:
    cur.execute("SELECT chunk_id, document_id, source, page, text, chapter, section, content_type "
                "FROM chunks WHERE strategy = 'B' ORDER BY chunk_id")
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
chunks = [{**{k: r[k] for k in ("chunk_id", "document_id", "source", "page", "text")},
           "metadata": {k: r[k] for k in ("chapter", "section", "content_type")}} for r in rows]
Path("data").mkdir(exist_ok=True)
Path("data/day6_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
print(len(chunks), "chunks exported")