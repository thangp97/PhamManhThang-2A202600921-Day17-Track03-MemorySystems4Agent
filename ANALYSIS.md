# Phân tích kết quả benchmark — Memory Systems for AI Agent

> Số liệu dưới đây lấy từ `python benchmark.py` (chạy trong `src/`). Cả hai agent
> dùng đường **offline tất định** (`force_offline=True`) nên kết quả tái lập được,
> không phụ thuộc API key.

## 1. Bảng kết quả

### Standard Benchmark (`data/conversations.json`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2138 | 16698 | 0.11 | 0.61 | 0 | 0 |
| Advanced | 2956 | 24549 | 0.50 | 0.86 | 158 | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2466 | 39290 | 0.00 | 0.50 | 0 | 0 |
| Advanced | 337 | 24538 | 0.33 | 0.83 | 104 | 24 |

## 2. Vì sao Advanced có recall tốt hơn Baseline?

Câu hỏi recall luôn được hỏi ở **một thread mới** (`{id}-recall`), tức là một
"session" khác với lúc người dùng cung cấp fact.

- **Baseline** chỉ giữ short-term memory trong đúng `thread_id`. Sang thread mới,
  session rỗng → không có gì để tra cứu → trả về câu echo vô nghĩa
  (`"Mình đã ghi nhận: ..."`). Recall gần như bằng 0 (0.11 ở standard, **0.00** ở
  long-context).
- **Advanced** bóc fact ổn định bằng `extract_profile_updates()` và ghi bền vững
  vào `User.md` (`_persist_facts`). Sang thread mới, nó vẫn đọc lại đúng file
  profile của `user_id` đó → trả lời được. Recall tăng lên **0.50** (standard).

Đây chính là tác dụng của lớp **persistent memory**: nó tách "trí nhớ" ra khỏi
vòng đời của một thread.

## 3. Vì sao Advanced lại *tốn hơn* Baseline ở hội thoại ngắn?

Ở Standard Benchmark, Advanced tốn **nhiều token hơn**:
- Agent tokens: 2956 so với 2138
- Prompt tokens: 24549 so với 16698
- Compactions = **0** ở cả hai (thread chưa đủ dài để vượt ngưỡng 800 token).

Lý do: mỗi lượt, Advanced kéo theo phần ngữ cảnh cố định mà Baseline không có —
nội dung `User.md` + summary + message gần nhất (`_estimate_prompt_context_tokens`).
Ở thread ngắn, lịch sử của Baseline còn nhỏ, nên **chi phí cố định của lớp
persistent memory lấn át**: bỏ tiền ra nuôi memory nhưng chưa kịp thu lời.

Kết luận quan trọng: **compact memory không phải lúc nào cũng thắng**. Khi hội
thoại ngắn và compaction chưa kích hoạt, hệ memory phức tạp chỉ thêm overhead.

## 4. Vì sao compact giúp Advanced thắng ở hội thoại dài?

Ở Long-Context Stress Benchmark, bức tranh đảo ngược:
- Prompt tokens: Advanced **24538 < 39290** của Baseline.
- Compactions = **24** (compact memory đã nén lịch sử 24 lần).

Cơ chế: Baseline kéo lại **toàn bộ** lịch sử thread mỗi lượt → chi phí prompt
tăng tuyến tính theo độ dài, cộng dồn thành **bậc hai** qua cả thread. Advanced,
khi vượt ngưỡng, đẩy message cũ vào summary và `summarize_messages()` chỉ giữ
phần gần nhất — tức là **chủ động bỏ bớt** ngữ cảnh cũ → prompt mỗi lượt bị chặn
trên thay vì phình vô hạn.

Lưu ý đúng bản chất: compact tối ưu **`Prompt tokens processed`** (ngữ cảnh phải
xử lý lại mỗi lượt), **không** phải `Agent tokens only`. Con số `Agent tokens` của
Advanced thấp bất thường (337) chủ yếu vì reply offline ngắn và không lặp lại
message đầu vào như Baseline — đây là chi tiết của benchmark offline, không phải
thành tích của compact; cột cần nhìn để đánh giá compact là cột prompt.

## 5. Memory file tăng trưởng ra sao và rủi ro gì?

- `Memory growth` là 158 và 104 bytes — nhỏ vì dataset ít fact và `_persist_facts`
  **upsert theo field** (correction mới ghi đè fact cũ cùng `key`, không nhân đôi).
- Rủi ro khi mở rộng:
  1. **File phình to**: nếu mỗi câu đều sinh fact mới, `User.md` lớn dần và lại
     trở thành chi phí prompt cố định — đúng vấn đề mà compact đang cố tránh.
  2. **Lưu sai fact**: regex trong `extract_profile_updates()` có thể bắt nhầm.
     Hiện đã có guardrail bỏ qua câu hỏi (`?`, "là gì", "ở đâu"...) để không lưu
     fact từ câu người dùng đang *hỏi lại*.
  3. **Mất thông tin do nén**: `summarize_messages()` chỉ giữ vài message gần
     nhất của phần cũ. Điều này giải thích vì sao recall của Advanced ở
     long-context chỉ đạt **0.33** (thấp hơn 0.50 ở standard): một số fact bị nén
     mất trước khi kịp vào `User.md`. Đây là trade-off cốt lõi — nén giảm token
     nhưng có thể giảm độ chính xác.

## 6. Câu chuyện tổng thể (đúng luồng Rubric)

1. Baseline không nhớ dài hạn → recall ~0 ở thread mới.
2. Advanced thêm `User.md` → recall và quality tăng rõ.
3. Hội thoại dài làm prompt cost của Baseline tăng mạnh (39290).
4. Compact memory kéo `Prompt tokens processed` của Advanced xuống (24538).
5. Hệ mạnh hơn nhưng phức tạp hơn: tốn hơn ở thread ngắn, và cần guardrail tốt
   (confidence khi ghi, chống phình file, tránh nén mất fact).

## 7. Kiểm chứng bằng test

`src/test_agents.py` khóa lại đúng các kết luận trên:
- `test_cross_session_recall` — Advanced nhớ qua thread mới, Baseline thì không.
- `test_compact_trigger` — thread dài thật sự kích hoạt nén.
- `test_compact_reduces_prompt_load_on_long_thread` — prompt của Advanced (đã nén)
  thấp hơn Baseline (phình) ở cả lượt cuối lẫn tổng tích lũy.
- `test_user_markdown_read_write_edit` — vòng đời `User.md` (read/write/edit/size).
