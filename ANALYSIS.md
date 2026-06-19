# Phân tích kết quả benchmark — Memory Systems for AI Agent

> Số liệu dưới đây lấy từ `python benchmark.py` (chạy trong `src/`). Cả hai agent
> dùng đường **offline tất định** (`force_offline=True`) nên kết quả tái lập được,
> không phụ thuộc API key.

## 1. Bảng kết quả

### Standard Benchmark (`data/conversations.json`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2138 | 16698 | 0.11 | 0.61 | 0 | 0 |
| Advanced | 2686 | 28330 | 0.50 | 0.86 | 353 | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2466 | 39290 | 0.00 | 0.50 | 0 | 0 |
| Advanced | 396 | 25593 | 1.00 | 1.00 | 290 | 24 |

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
- Agent tokens: 2686 so với 2138
- Prompt tokens: 28330 so với 16698
- Compactions = **0** ở cả hai (thread chưa đủ dài để vượt ngưỡng 800 token).

Lý do: mỗi lượt, Advanced kéo theo phần ngữ cảnh cố định mà Baseline không có —
nội dung `User.md` + summary + message gần nhất (`_estimate_prompt_context_tokens`).
Ở thread ngắn, lịch sử của Baseline còn nhỏ, nên **chi phí cố định của lớp
persistent memory lấn át**: bỏ tiền ra nuôi memory nhưng chưa kịp thu lời.

Kết luận quan trọng: **compact memory không phải lúc nào cũng thắng**. Khi hội
thoại ngắn và compaction chưa kích hoạt, hệ memory phức tạp chỉ thêm overhead.

## 4. Vì sao compact giúp Advanced thắng ở hội thoại dài?

Ở Long-Context Stress Benchmark, bức tranh đảo ngược:
- Prompt tokens: Advanced **25593 < 39290** của Baseline.
- Compactions = **24** (compact memory đã nén lịch sử 24 lần).

Cơ chế: Baseline kéo lại **toàn bộ** lịch sử thread mỗi lượt → chi phí prompt
tăng tuyến tính theo độ dài, cộng dồn thành **bậc hai** qua cả thread. Advanced,
khi vượt ngưỡng, đẩy message cũ vào summary và `summarize_messages()` chỉ giữ
phần gần nhất — tức là **chủ động bỏ bớt** ngữ cảnh cũ → prompt mỗi lượt bị chặn
trên thay vì phình vô hạn.

Lưu ý đúng bản chất: compact tối ưu **`Prompt tokens processed`** (ngữ cảnh phải
xử lý lại mỗi lượt), **không** phải `Agent tokens only`. Con số `Agent tokens` của
Advanced thấp bất thường (396) chủ yếu vì reply offline ngắn và không lặp lại
message đầu vào như Baseline — đây là chi tiết của benchmark offline, không phải
thành tích của compact; cột cần nhìn để đánh giá compact là cột prompt.

## 5. Memory file tăng trưởng ra sao và rủi ro gì?

- `Memory growth` là 353 và 290 bytes — vẫn nhỏ vì `_persist_facts` **upsert theo
  field** (correction ghi đè fact cũ cùng `key`, không nhân đôi). So với bản chưa
  có bonus (158/104), kích thước tăng do mỗi dòng fact giờ kèm metadata
  (`conf`, `mentions`, `turn`) phục vụ confidence/decay — một chi phí có chủ đích.
- Rủi ro khi mở rộng:
  1. **File phình to**: nếu mỗi câu đều sinh fact mới, `User.md` lớn dần và lại
     trở thành chi phí prompt cố định — đúng vấn đề mà compact đang cố tránh.
  2. **Lưu sai fact**: regex trong `extract_profile_updates()` có thể bắt nhầm.
     Ngoài guardrail bỏ qua câu hỏi, phần **Bonus** (mục 8) bổ sung confidence
     threshold để loại câu nhiễu/đùa.
  3. **Mất thông tin do nén**: `summarize_messages()` chỉ giữ vài message gần
     nhất của phần cũ → nén có thể cắt mất chi tiết. Điểm mấu chốt: nhờ lớp
     **persistent** bóc fact ổn định vào `User.md` *trước/độc lập* với việc nén,
     recall của Advanced ở long-context đạt **1.00** dù compact đã chạy 24 lần.
     Nói cách khác, compact làm mất *raw detail* nhưng *stable facts* vẫn an toàn
     trong `User.md` — đúng vai trò của việc tách hai lớp memory.

## 6. Câu chuyện tổng thể (đúng luồng Rubric)

1. Baseline không nhớ dài hạn → recall ~0 ở thread mới.
2. Advanced thêm `User.md` → recall và quality tăng rõ.
3. Hội thoại dài làm prompt cost của Baseline tăng mạnh (39290).
4. Compact memory kéo `Prompt tokens processed` của Advanced xuống (25593).
5. Hệ mạnh hơn nhưng phức tạp hơn: tốn hơn ở thread ngắn, và cần guardrail tốt
   (confidence khi ghi, chống phình file, tránh nén mất fact).

## 7. Kiểm chứng bằng test

`src/test_agents.py` (8 test, đều pass) khóa lại đúng các kết luận trên:
- `test_cross_session_recall` — Advanced nhớ qua thread mới, Baseline thì không.
- `test_compact_trigger` — thread dài thật sự kích hoạt nén.
- `test_compact_reduces_prompt_load_on_long_thread` — prompt của Advanced (đã nén)
  thấp hơn Baseline (phình) ở cả lượt cuối lẫn tổng tích lũy.
- `test_user_markdown_read_write_edit` — vòng đời `User.md` (read/write/edit/size).
- 4 test bonus (xem mục 8): entity extraction, confidence threshold, conflict
  handling, memory decay/reinforcement.

## 8. Bonus: confidence + conflict + entity + decay

Bốn mở rộng dưới đây được thiết kế quanh `data/advanced_long_context.json` — bộ
dữ liệu cố tình cài bẫy: đính chính (Huế → Đà Nẵng), nhiễu ("Hà Nội chỉ là đi
họp", "product manager chỉ là câu đùa"), fact ghép nhiều từ, và fact lặp lại
nhiều lần. Kết quả: recall long-context tăng từ **0.33 → 1.00**, quality
**0.83 → 1.00**, mà vẫn giữ prompt thấp hơn Baseline (25593 < 39290).

| Bonus | Giải quyết vấn đề gì | Cải thiện recall/cost ra sao | Rủi ro mới |
|---|---|---|---|
| **Entity extraction** (`extract_profile_updates` chấm theo clause, bắt giá trị 1–3 token, cho cả chữ số) | Regex cũ bỏ sót tên ghép ("DũngCT Stress"), nghề ("MLOps engineer"), style có số ("3 bullet") | Đưa đúng 4 fact ổn định vào `User.md` → recall long-context lên 1.00 | Vẫn là heuristic; ngôn ngữ tự nhiên ngoài mẫu có thể trượt |
| **Confidence threshold** (`_score` + `confidence_threshold=0.6`) | Câu nhiễu/đùa làm bẩn profile ("Hà Nội", "product manager") | Marker phủ định ("không phải", "chỉ là", "đùa", "lúc đầu") hạ tin cậy ×0.25 → rơi dưới ngưỡng → KHÔNG ghi | Ngưỡng quá cao có thể bỏ sót fact thật mơ hồ |
| **Conflict handling** (so `confidence` mới với tin cậy hiệu dụng đã decay của fact cũ) | Đính chính phải cập nhật và **không** giữ giá trị cũ sai | Huế bị thay bằng Đà Nẵng; profile không chứa đồng thời hai nơi ở | Correction "yếu" hơn fact cũ chưa phai sẽ bị từ chối — cần dữ liệu/decay hợp lý |
| **Memory decay + reinforcement** (`ProfileFact.effective_confidence`) | Fact cũ không nên "bất tử"; fact nhắc lại nhiều nên ưu tiên | Tin cậy giảm theo lượt im lặng, tăng theo `mentions`; dùng để xếp hạng khi recall mở và làm "hàng rào" cho conflict | Dùng đồng hồ logic (turn) chứ không phải thời gian thực; cần chọn `decay_rate` cẩn thận |

Cài đặt: `confidence_threshold` và `decay_rate` nằm trong `LabConfig`
(override qua env `MEMORY_CONFIDENCE_THRESHOLD`, `MEMORY_DECAY_RATE`). Metadata
mỗi fact lưu ngay trong `User.md` dưới dạng HTML comment nên file vẫn đọc được
như markdown thường.
