# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `day6_retriever.py` | Dense retriever: import `MyEmbedder` và `PgVectorStore` từ repo Day 6, trả chunk theo schema Day 7 |
| `day6_export.py` | Xuất chunk strategy B từ pgvector Day 6 ra `data/day6_chunks.json` |
| `make_cases.py` | Gộp câu có đáp án và câu thiếu evidence thành `data/cases.json` (không chứa gold) |
| `eval_retrieval.py` | Benchmark retrieval: `tfidf`, `dense`, `dense_filter`, `hybrid` (RRF) |
| `llm_controller.py` | `Controller.decide` dùng LLM thật qua API tương thích OpenAI, một tool `search` |
| `run_agent_bench.py` | Chạy agent trên một split, ghi `traces.json` cho `agent_metrics.py` |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Bài này dùng repo Day 6 `K5-L3A-Day06-TranThanhLong-2A202603674-DataIngestionEngineering`, nhánh **`feature/paddle-ocr`** (full book + OCR hình), đặt cạnh repo Day 7:

| Thành phần Day 6 | Giá trị |
|---|---|
| Tài liệu | OpenStax *Anatomy and Physiology 2e*, bản đầy đủ (CC BY 4.0) |
| Vector store | PostgreSQL 16 + pgvector 0.8.7 (Docker, project `day6_student_starter`, cổng `127.0.0.1:55433`, bảng `chunks`) |
| Embedding | Ollama `qwen3-embedding:4b`, 2560 chiều; query có instruction Qwen3 |
| Index dùng ở Day 7 | strategy `B` (chunk theo section, ≤ 500 ký tự): 7768 chunk (6675 `text` + 1093 `figure_ocr`) |
| Câu hỏi | `benchmark/questions_full_book.json` (34 câu: 24 dev, 10 test) |

**Bước 1 – Khởi động Day 6.** Trong thư mục `starter/` của repo Day 6 (luôn dùng đúng `-p`, vì volume dữ liệu gắn với tên project):

```bash
docker compose -p day6_student_starter up -d --wait
ollama list   # phải có qwen3-embedding:4b
```

Giữ repo Day 6 ở nhánh `feature/paddle-ocr`: `day6_retriever.py` import code Day 6 trực tiếp, và `store.py` ở `main` không trả cột `content_type`.

**Bước 2 – Tạo `.env`** ở thư mục gốc Day 7 từ [`.env.example`](../.env.example) (`DATABASE_URL`, `EMBEDDING_MODEL`, `DAY6_STARTER`, `LLM_*`). Không commit `.env`.

**Bước 3 – Kiểm tra kết nối dense** (chạy trong `starter/`, đã kích hoạt `.venv`):

```bash
python -c "from day6_retriever import Day6Retriever; print([(r['id'], r['page'], round(r['score'], 3)) for r in Day6Retriever().search('What is human anatomy?')])"
```

**Bước 4 – Xuất và chuyển dữ liệu** (từ thư mục gốc Day 7):

```bash
python starter/day6_export.py
python starter/convert_day6.py --chunks data/day6_chunks.json --questions ../K5-L3A-Day06-TranThanhLong-2A202603674-DataIngestionEngineering/benchmark/questions_full_book.json --output data/converted
python starter/make_cases.py
```

- `day6_export.py` đọc strategy B từ pgvector → `data/day6_chunks.json`, gồm `metadata.page_end`, `chapter`, `section`, `content_type`.
- `convert_day6.py` đổi tên trường (`chunk_id`→`id`, `document_id`→`source_id`, `gold_source_id/gold_page/evidence_quote`→`gold`) → `data/converted/`. Trường `filter` (ví dụ `{"chapter": 2}`) của câu hỏi Day 6 được giữ nguyên.
- `make_cases.py` gộp `data/converted/questions.json` và `data/unanswerable.json` → `data/cases.json` (không chứa gold).

Một số chunk Day 6 trải qua hai trang (`page` = trang đầu, `metadata.page_end` = trang cuối); `hit()` trong `baseline.py` chấp nhận trang gold nằm trong khoảng này.

**Bước 5 – Benchmark retrieval trên dev:**

```bash
python starter/eval_retrieval.py --method tfidf --split dev --output starter/results/dev-tfidf
python starter/eval_retrieval.py --method dense --split dev --output starter/results/dev-dense
python starter/eval_retrieval.py --method hybrid --split dev --output starter/results/dev-hybrid
python starter/eval_retrieval.py --method dense_filter --split dev --output starter/results/dev-dense-filter
python starter/eval_retrieval.py --method dense --split dev --budget 2000 --output starter/results/dev-dense-b2000
```

`dense_filter` chỉ áp dụng filter có sẵn trong câu hỏi (do người dùng cung cấp), không suy ra từ gold.

Cải thiện thêm:

```bash
# Hybrid có trọng số: RRF với dense = 1.0, TF-IDF = 0.3
python starter/eval_retrieval.py --method hybrid --lexical-weight 0.3 --split dev --output starter/results/dev-hybrid-w03
# Rerank: dense lấy 20 candidate, cross-encoder xếp lại, giữ top-3 (cần: pip install sentence-transformers)
python starter/eval_retrieval.py --method dense_rerank --split dev --output starter/results/dev-dense-rerank
```

`dense_rerank` mặc định dùng `cross-encoder/ms-marco-MiniLM-L-6-v2` (tải từ Hugging Face lần đầu, chạy CPU); đổi bằng `--rerank-model`. Trace giữ `retrieval_rank` và `retrieval_score` của dense để so sánh trước/sau rerank.

**Bước 6 – Chạy agent:**

```bash
python starter/run_agent_bench.py --method dense --split dev --budget 2000 --output starter/results/agent-dev-dense
```

Đối chiếu lại nguồn, trang và evidence trong trace; khớp nguyên văn bằng code không thay thế đánh giá thủ công.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
