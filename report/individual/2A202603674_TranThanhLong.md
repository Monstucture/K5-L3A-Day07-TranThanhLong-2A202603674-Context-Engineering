# Báo cáo Day 7

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** OpenStax *Anatomy and Physiology 2e* (bản đầy đủ, `anatomy-and-physiology-2e.pdf`), giấy phép CC BY 4.0. 1029 trang có nội dung được index, 318 trang bị loại theo manifest Day 6 (log: `starter/results_ocr_full_book/ingestion_log.json` trong repo Day 6). Dùng index **strategy B** của Day 6 (chunk theo section, tối đa 500 ký tự): **7768 chunk**, gồm 6675 chunk text và 1093 chunk `figure_ocr` (text OCR từ hình). 544 chunk trải qua 2 trang (`page` → `page_end`).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8.7 (Docker image `pgvector/pgvector:0.8.7-pg16`, project `day6_student_starter`, bảng `chunks`). Tìm kiếm cosine chính xác (không dùng ANN index). Embedding: Ollama `qwen3-embedding:4b`, 2560 chiều; query được thêm instruction của Qwen3. Code Day 6 lấy từ nhánh **`feature/paddle-ocr`** và được import trực tiếp qua `DAY6_STARTER` (`starter/day6_retriever.py`), không sao chép.
  - **Phiên bản index:** Day 6 commit `e3bbd72`, ingestion chạy lại lúc 2026-11-14 08:05 UTC. Ở phiên bản này chunk strategy B được embed kèm tiêu đề chương/section (`with_heading` trong `chunking.py`); text lưu trong DB không đổi (đã kiểm tra: 7768/7768 chunk giống hệt file export). Mọi kết quả dense trong bảng dưới dùng phiên bản index này (các lần chạy trước khi re-index đã được chạy lại).
- **Agent model, cách chạy và phân chia dev/test:** Controller gọi model qua API tương thích OpenAI (`starter/llm_controller.py`): **OpenAI `gpt-4o-mini`** (`https://api.openai.com/v1`), `temperature=0`, `max_tokens=512`, một tool `search`. Cả 4 lần chạy agent (TF-IDF/dense × dev/test) dùng cùng model này. Trước đó đã thử OpenRouter `openrouter/auto` (hết credit, lỗi 402; model được định tuyến tự động nên không cố định) và Gemini `gemini-2.5-flash` (lỗi 404, đã ngừng cho người dùng mới); kết quả các lần thử này không được dùng. Benchmark (`data/cases.json`, 42 câu): dev = 24 câu có đáp án + 2 câu thiếu evidence; test = 10 câu có đáp án + 6 câu thiếu evidence. Câu có đáp án lấy từ benchmark Day 6 (`benchmark/questions_full_book.json`, đã chia dev/test từ Day 6). Phân loại: dev gồm 8 direct, 8 comparison, 4 paraphrase, 3 metadata_filter, 1 condition; test gồm 3 direct, 3 comparison, 2 paraphrase, 2 metadata_filter.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** Câu có đáp án có `gold` = `{source_id, page, evidence}`; một chunk được tính là **hit** khi đúng `source_id`, trang gold nằm trong khoảng `page..page_end` của chunk, và câu evidence nằm **nguyên văn** trong text của chunk (`hit()` trong `starter/baseline.py`, tương đương `evaluate_query` của Day 6). Câu thiếu evidence nằm trong `data/unanswerable.json`. Gold không được đưa vào agent hay retrieval (`cases.json` không chứa gold). Filter `chapter` chỉ dùng khi câu hỏi có sẵn filter do người dùng cung cấp (3 câu dev, 2 câu test).
  - *TODO / giới hạn:* `absence_reason` của các câu thiếu evidence vẫn là placeholder ("Searched the PDF for X and Y"), và lý do của u6 không khớp câu hỏi. Phần lớn câu (u1–u3, u5–u8) không gần chủ đề sách nên kiểm tra abstention còn dễ; chỉ u4 ("chapter 33", sách có 28 chương) là câu gần chủ đề. Cần thay bằng câu gần chủ đề và ghi cách đã kiểm tra trong PDF.

## Retrieval và thử nghiệm

Tất cả chạy bằng `starter/eval_retrieval.py` trên cùng corpus (7768 chunk strategy B), cùng 24 câu dev có đáp án, cùng định nghĩa hit, top-3. Context budget đo bằng **ký tự** (gồm cả header nguồn). Thời gian là median mỗi query trên máy local (dense gồm embedding query qua Ollama; rerank gồm thêm cross-encoder trên CPU). Chi phí dịch vụ = 0 USD cho mọi cấu hình (embedding, DB, cross-encoder đều chạy local; cross-encoder tải ~90 MB một lần từ Hugging Face).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF (lexical), top-3, không filter, không rerank, 800 ký tự | Baseline lexical của starter | dev | 6/24 (25,0%) | 9/24 (37,5%) | 6/24 (25,0%) | 15,6 ms / 0 USD |
| Phương pháp 2 | Dense trên pgvector Day 6 (qwen3-embedding:4b), top-3, không filter, 800 ký tự | Embedding hiểu paraphrase tốt hơn khớp từ khóa | dev | 17/24 (70,8%) | 19/24 (79,2%) | 18/24 (75,0%) | 88,8 ms / 0 USD |
| Phương pháp 3 | Hybrid: RRF (k=60) giữa dense top-20 và TF-IDF top-20, trọng số 1:1, top-3, 800 ký tự | Kết hợp dense và lexical sẽ bắt thêm câu chứa thuật ngữ chính xác | dev | 9/24 (37,5%) | 15/24 (62,5%) | 10/24 (41,7%) | 97,6 ms / 0 USD |
| Cải thiện 1 | Hybrid có trọng số: RRF giữa dense top-20 (trọng số 1,0) và **BM25** top-20 (trọng số 0,3), top-3, 800 ký tự | BM25 là lexical tốt hơn TF-IDF; trọng số 0,3 để lexical chỉ bổ sung, không lấn át dense → ít nhất bằng dense | dev | 14/24 (58,3%) | 17/24 (70,8%) | 14/24 (58,3%) | 93,5 ms / 0 USD |
| Cải thiện 2 | Rerank: dense lấy 20 candidate, cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` xếp lại, top-3, 800 ký tự | Cross-encoder đọc cả câu hỏi và chunk nên đưa chunk evidence lên trên chunk "Learning Objectives" và chunk nhãn hình (q13, q17) | dev | 18/24 (75,0%) | 20/24 (83,3%) | 18/24 (75,0%) | 221,6 ms / 0 USD |
| Cấu hình đã chọn (dev) | Dense, top-3, không filter, không rerank, **2000 ký tự** | Budget 2000 giữ đủ top-3 trong context; với tài liệu này dense là lựa chọn hiệu quả về chi phí (xem "Lý do chọn cấu hình") | dev | 17/24 (70,8%) | 19/24 (79,2%) | 19/24 (79,2%) | 87,8 ms / 0 USD |
| Cấu hình đã chọn | Dense, top-3, không filter, không rerank, 2000 ký tự | Chốt trên dev trước khi chạy test | test | 6/10 (60,0%) | 10/10 (100%) | 10/10 (100%) | 75,5 ms / 0 USD |
| Baseline | TF-IDF, top-3, 800 ký tự | So sánh với cấu hình đã chọn | test | 3/10 (30,0%) | 5/10 (50,0%) | 3/10 (30,0%) | 15,0 ms / 0 USD |

**Kết quả test (retrieval):** dense 2000 tìm được evidence trong top-3 và đưa vào context ở **cả 10/10** câu test có đáp án, so với 5/10 và 3/10 của TF-IDF. Hit@1 chỉ 6/10: ở q05, q09, q30, q31 evidence nằm ở hạng 2–3, nhưng budget 2000 vẫn đưa vào context. Kết quả test cao hơn dev (Context Evidence 19/24) một phần vì test chỉ có 10 câu và không có câu loại q23/q24/q33 (so sánh nhiều khái niệm) như dev; không chỉnh cấu hình sau khi xem test.

**Các thử nghiệm bổ sung trên dev** (cùng điều kiện):

| Cấu hình | Giả thuyết / thay đổi | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian |
|---|---|---|---|---|---|
| BM25 (lexical, k1=1,5, b=0,75), 800 ký tự | BM25 bão hòa từ lặp và chuẩn hóa độ dài chunk → tốt hơn TF-IDF | 7/24 (29,2%) | 14/24 (58,3%) | 7/24 (29,2%) | 6,9 ms |
| Hybrid TF-IDF, trọng số lexical 0,3 | Giảm trọng số TF-IDF để chunk sai của TF-IDF không đẩy chunk đúng của dense xuống | 12/24 (50,0%) | 15/24 (62,5%) | 13/24 (54,2%) | 118,4 ms |
| Dense + filter `chapter` (do người dùng cung cấp) | Lọc theo chương loại bỏ chunk ngoài chủ đề → tăng Hit@1 | 17/24 (70,8%) | 19/24 (79,2%) | 18/24 (75,0%) | 76,0 ms |
| Dense, budget **2000** ký tự | Budget 800 chỉ chứa ~1 chunk nên evidence hạng 2–3 bị cắt; 2000 giữ đủ top-3 | 17/24 (70,8%) | 19/24 (79,2%) | 19/24 (79,2%) | 87,8 ms |
| Rerank, budget **2000** ký tự | Như trên, áp dụng cho rerank (evidence hạng 2 của q18, q20 bị cắt ở budget 800) | 18/24 (75,0%) | 20/24 (83,3%) | 20/24 (83,3%) | 255,6 ms |

*Ghi chú:* lần chạy dense budget 2000 đầu tiên (trên index cũ, trước khi Day 6 được re-index lúc 15:05) cho 18/20/20; đã chạy lại trên index hiện tại và chỉ dùng kết quả mới.

**Nhận xét từng thử nghiệm:**

- **Dense vs TF-IDF:** dense tăng Hit@1 từ 6/24 lên 17/24. TF-IDF trượt hầu hết câu paraphrase và comparison (q08, q10, q11, q20–q22, q34) vì câu hỏi không dùng đúng từ trong sách.
- **BM25 vs TF-IDF:** BM25 tốt hơn TF-IDF rõ ở Hit@3 (14/24 so với 9/24; thêm q12, q14, q21, q22, q28) nhờ chuẩn hóa độ dài chunk, nhưng Hit@1 gần như không đổi (7 so với 6). Lexical vẫn kém xa dense.
- **Hybrid (phương pháp 3) kém hơn dense:** Hit@1 9/24, Hit@3 15/24. Với trọng số 1:1, chunk sai của TF-IDF đẩy chunk đúng của dense xuống (q08, q10, q11, q34 mất khỏi top-3). Giả thuyết **không đúng**.
- **Cải thiện 1 (hybrid BM25, trọng số 0,3) – giả thuyết không được xác nhận:** tốt hơn nhiều so với hybrid 1:1 (Hit@1 14 so với 9; Hit@3 17 so với 15) và tốt hơn hybrid TF-IDF trọng số 0,3 (14/17 so với 12/15), nên cả hai thay đổi (BM25 và giảm trọng số) đều có ích. Nhưng vẫn **thấp hơn dense** (14/17 so với 17/19): q08, q34 vẫn mất khỏi top-3, q10, q11, q20 rơi khỏi top-1. Trên corpus này lexical không mang lại câu nào mà dense bỏ sót.
- **Cải thiện 2 (rerank) – giả thuyết đúng một phần:** rerank **không làm câu nào kém đi** so với dense, và thêm 1 câu Hit@1 (**q11**: dense xếp chunk nhãn hình `p032:fig134` lên hạng 1, rerank đưa evidence `s1.3:c001` từ hạng 2 lên hạng 1) và 1 câu Hit@3 (**q18**: evidence `s1.6:c014` từ hạng 6 của dense lên hạng 2). Nhưng hai câu trong giả thuyết không thay đổi: **q13** đã đúng ở dense trên index mới (embedding kèm tiêu đề section), và **q17** vẫn sai vì cross-encoder cũng chấm chunk nhãn hình `p041:fig160` cao nhất (5,09). Context Evidence không tăng (18/24) vì budget 800 cắt evidence ở hạng 2 (q18, q20). Đánh đổi: chậm hơn ~2,5 lần (221,6 so với 88,8 ms) và thêm dependency `sentence-transformers` (PyTorch).
- **Filter chương:** kết quả giống hệt dense. 3 câu dev có filter (q12, q14, q26) đã đúng ở top-1 khi không có filter.
- **Budget 2000 – giả thuyết được xác nhận:** budget 800 chỉ chứa trung bình 1,42 chunk (dense) và 1,12 chunk (rerank); budget 2000 giữ cả 3 chunk (trung bình 1369 ký tự với dense, 1491 với rerank). Context Evidence tăng: dense 18 → 19/24 (q20 vào context), rerank 18 → 20/24 (q18, q20 vào context), Context Evidence bằng Hit@3 ở cả hai. Hit@1/Hit@3 không đổi (ranking giữ nguyên), thời gian gần như không đổi. Đánh đổi: context gửi model dài hơn ~2,3–2,8 lần → tốn token hơn khi chạy agent.

**Hai lỗi dev, nguyên nhân và cách thử (index hiện tại):**

1. **q17** – *"Differentiate between anterior, posterior, superior, …, proximal, and distal."* (gold trang 40). Ở dense, rerank và mọi hybrid, hạng 1 là chunk **`figure_ocr`** `p041:fig160:c00` (text OCR của hình 1.13, chỉ là danh sách nhãn "Superior Cranial Right Left…"). Chunk này chứa đúng các từ trong câu hỏi nhưng không có câu giải thích, nên cả embedding lẫn cross-encoder đều chấm cao. Chunk evidence `s1.6:c008` không vào top-3 ở index hiện tại. **Đã thử:** rerank → vẫn sai. **Hướng tiếp theo:** hạ điểm hoặc loại chunk `figure_ocr` chỉ gồm nhãn ngắn; lấy thêm chunk lân cận cùng section.
2. **q20** – *"From most to least common, which four elements make up most of the human body?"* (gold trang 58). Dense xếp chunk nhãn hình `p058:fig207:c00` lên hạng 1; evidence `s2.1:c006` ở hạng 2. Rerank loại được chunk nhãn hình nhưng xếp `s2.1:c007` (đoạn kế tiếp) lên trên evidence. Ở cả hai, budget 800 chỉ chứa chunk hạng 1 nên **evidence nằm trong top-3 nhưng không vào context**. **Đã thử:** budget 2000 → evidence vào context ở cả dense và rerank (Context Evidence của q20 đạt). Hit@1 vẫn sai; hướng tiếp theo là hạ điểm chunk nhãn hình (dense) hoặc lấy thêm chunk lân cận.

**Lỗi còn lại:** q23, q24, q33 sai ở mọi cấu hình; q18 chỉ rerank lấy được. Đều là câu so sánh/liệt kê nhiều khái niệm. Retriever trả đúng section và đúng trang, nhưng câu evidence gold là **một câu cụ thể** nằm ở chunk khác cùng section. Một phần là giới hạn của metric (khớp nguyên văn một câu evidence cho câu hỏi cần nhiều đoạn), một phần là câu hỏi tổng hợp cần nhiều chunk hơn top-3. Hướng thử: nhiều lượt search trong agent (tách từng khái niệm), thêm chunk lân cận, hoặc mở rộng gold thành nhiều evidence thay thế.

**Lý do chọn cấu hình:** tiêu chí công bố trước: ưu tiên Context Evidence Rate, sau đó Hit@3, Hit@1, rồi thời gian/chi phí.

- Trên index hiện tại với budget 800: rerank và dense **hòa** Context Evidence (18/24); rerank hơn 1 câu ở Hit@3 (20 so với 19) và Hit@1 (18 so với 17), nhưng chậm hơn ~2,5 lần. Chênh lệch chỉ 1/24 câu nên bằng chứng còn yếu.
- Budget là yếu tố tác động lớn nhất lên Context Evidence, nên so sánh cuối cùng ở budget 2000: **rerank 2000** đạt Hit@1 18/24, Hit@3 20/24, Context Evidence 20/24, 255,6 ms; **dense 2000** đạt 17/24, 19/24, 19/24, 87,8 ms.
- Theo đúng thứ tự tiêu chí, rerank 2000 đứng đầu (hơn 1 câu ở cả ba chỉ số). **Quyết định: chọn dense 2000**, đặt chi phí/thời gian lên trước, vì:
  - chênh lệch chỉ **1/24 câu** (~4%) ở mỗi chỉ số, trong phạm vi dao động đã thấy khi index thay đổi (dense trên index cũ và mới lệch 1 câu), nên chưa đủ chứng minh rerank tốt hơn trên tài liệu này;
  - rerank chậm hơn **~2,9 lần** (255,6 so với 87,8 ms) và cần thêm model cross-encoder cùng dependency `sentence-transformers` (PyTorch);
  - với tài liệu này (một sách giáo khoa, embedding đã kèm tiêu đề section), dense đã đưa evidence vào context ở 19/24 câu.
- Rerank vẫn là lựa chọn đáng cân nhắc cho kho tài liệu lớn (ví dụ tài liệu doanh nghiệp nhiều nguồn), nơi số candidate gần giống nhau nhiều hơn và lợi ích xếp hạng lại có thể lớn hơn chi phí.
- Đây là ngoại lệ so với tiêu chí công bố trước, được ghi rõ ở đây. Cấu hình được chốt trên dev **trước khi chạy test**; không chỉnh thêm sau khi xem kết quả test.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** Agent dùng khung `run_agent` trong `starter/agentic.py` với controller `LLMController` (`starter/llm_controller.py`). Một tool `search(query)` gọi retriever đã chọn. Tối đa 4 lượt search (`--max-searches 4`), top-3 mỗi lượt; context là các chunk tích lũy qua các lượt, loại trùng ID, cắt theo budget ký tự (`pack_context`). Model phải trả lời kèm citation `[chunk-id]` hoặc trả `INSUFFICIENT_EVIDENCE`. `run_agent` chuyển thành **abstain** khi: không có citation, citation không nằm trong context, hoặc hết số lượt search. Lỗi model/vector store được ghi `status: "error"` kèm thông báo lỗi trong trace (`starter/run_agent_bench.py`), không thay bằng câu trả lời giả. Nội dung tài liệu được chỉ định là dữ liệu, không phải chỉ dẫn (system prompt).
- **Lần chạy hiện tại:** lần chạy dev đầu tiên lỗi ngay ở q01: `APIStatusError 402 – This request requires more credits, or fewer max_tokens` (OpenRouter giữ chỗ max_tokens mặc định lớn hơn số credit). Đã thêm `max_tokens=512` (commit `3a75587`). *TODO – cần chạy:* agent trên dev cho baseline TF-IDF và cấu hình đã chọn (dense, budget 2000).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline TF-IDF / dev | *TODO – cần chạy agent + chấm nhãn* | | | | | | |
| Dense 2000 (đã chọn) / dev | *TODO – cần chạy agent + chấm nhãn* | | | | | | |
| Baseline TF-IDF / test | *TODO – cần chạy agent + chấm nhãn* | | | | | | |
| Dense 2000 (đã chọn) / test | *TODO – cần chạy agent + chấm nhãn* | | | | | | |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| *TODO – cần chạy agent trên test và chấm tay* (≥5 câu test có đáp án + 2 câu thiếu evidence) | | | | |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Retrieval dev: `starter/results/dev-{tfidf,bm25,dense,dense-filter,dense-b2000,hybrid,hybrid-w03,hybrid-bm25-w03,dense-rerank,dense-rerank-b2000}/{summary,traces}.json`.
  - Lệnh (chạy từ thư mục gốc, đã kích hoạt `.venv`, Docker Day 6 và Ollama đang chạy):
    - `python starter/eval_retrieval.py --method tfidf --split dev --output starter/results/dev-tfidf`
    - `python starter/eval_retrieval.py --method bm25 --split dev --output starter/results/dev-bm25`
    - `python starter/eval_retrieval.py --method dense --split dev --output starter/results/dev-dense`
    - `python starter/eval_retrieval.py --method dense_filter --split dev --output starter/results/dev-dense-filter`
    - `python starter/eval_retrieval.py --method hybrid --split dev --output starter/results/dev-hybrid`
    - `python starter/eval_retrieval.py --method hybrid --lexical-weight 0.3 --split dev --output starter/results/dev-hybrid-w03`
    - `python starter/eval_retrieval.py --method hybrid_bm25 --lexical-weight 0.3 --split dev --output starter/results/dev-hybrid-bm25-w03`
    - `python starter/eval_retrieval.py --method dense_rerank --split dev --output starter/results/dev-dense-rerank`
    - `python starter/eval_retrieval.py --method dense --split dev --budget 2000 --output starter/results/dev-dense-b2000`
    - `python starter/eval_retrieval.py --method dense_rerank --split dev --budget 2000 --output starter/results/dev-dense-rerank-b2000`
  - Retrieval test (*TODO – cần chạy*): `--method tfidf --split test --output starter/results/test-tfidf` và `--method dense --split test --budget 2000 --output starter/results/test-dense-b2000`.
  - Agent (*TODO – cần chạy*): `python starter/run_agent_bench.py --method <tfidf|dense> --split <dev|test> [--budget 2000] --output starter/results/agent-<split>-<method>`; nhãn: `reviews.json` cùng thư mục; tổng hợp: `python starter/agent_metrics.py --cases data/cases.json --traces <traces.json> --reviews <reviews.json> --split <split> --output <summary.json>`.
  - Chuẩn bị dữ liệu: `starter/day6_export.py` → `data/day6_chunks.json`; `starter/convert_day6.py` → `data/converted/`; `starter/make_cases.py` → `data/cases.json`.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - Trước/sau: hybrid 1:1 TF-IDF → hybrid 0,3 BM25: Hit@1 9 → 14, Hit@3 15 → 17 (vẫn dưới dense 17/19). Dense → rerank: Hit@1 17 → 18 (q11), Hit@3 19 → 20 (q18), thời gian 88,8 → 221,6 ms. Budget 800 → 2000: Context Evidence dense 18 → 19 (q20), rerank 18 → 20 (q18, q20).
  - Re-index Day 6 giữa các lần chạy làm thay đổi kết quả dense (index cũ: Hit@1 18/24, Hit@3 20/24; index mới: 17/24, 19/24; q13 đúng hơn, q17 và q11 kém hơn). Mọi so sánh cuối cùng phải dùng cùng một phiên bản index.
  - Giới hạn: metric hit yêu cầu câu evidence nằm nguyên văn trong một chunk, nên đánh giá thấp câu so sánh cần nhiều đoạn (q18, q23, q24, q33). Chỉ có 24 câu dev nên chênh lệch 1 câu (~4%) chưa đủ kết luận chắc chắn. Benchmark dev/test được tạo từ Day 6; *TODO: xác nhận test Day 6 chưa từng dùng để chỉnh hệ thống* (nếu đã dùng, cần tạo test mới theo GUIDE.md). Chunk `figure_ocr` vừa giúp (thông tin trong hình) vừa gây nhiễu (nhãn hình ngắn có score cao: q11, q17, q20). Cần giữ repo Day 6 ở nhánh `feature/paddle-ocr`, commit `e3bbd72` để tái lập. Câu thiếu evidence hiện còn quá xa chủ đề (xem phần Dữ liệu).

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản: —
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý: —
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev: —

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | — | — | — |
| Ảnh, sơ đồ | — | — | — |
| Biểu đồ | — | — | — |
| Multi-hop | — | — | — |
| Text | — | — | — |
| Thiếu bằng chứng | — | — | — |
| **Toàn bộ** | — | — | — |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống: —
- Giới hạn, cách tái lập và đường dẫn benchmark: —
