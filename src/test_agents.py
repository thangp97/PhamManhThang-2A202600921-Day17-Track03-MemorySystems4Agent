from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Cấu hình cô lập cho test.

    - `state_dir` trỏ vào tmp_path nên mỗi test có thư mục profile riêng,
      không đụng vào `state/` thật của repo.
    - Hạ `compact_threshold_tokens` và `compact_keep_messages` thật thấp để
      compaction kích hoạt nhanh, không cần nạp hội thoại quá dài.
    """

    base = load_config(tmp_path)
    # Ngưỡng đủ cao để tích lũy nhiều message rồi mới nén: như vậy `summarize_messages`
    # thật sự BỎ bớt message cũ (lợi thế prompt của compact mới lộ ra), thay vì nén liên
    # tục từng lượt khiến summary phình theo. Vẫn thấp hơn nhiều so với mặc định (800)
    # để compaction kích hoạt trong phạm vi vài chục lượt của test.
    return replace(
        base,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=300,
        compact_keep_messages=4,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """`User.md` phải tạo, đọc, sửa được và báo đúng kích thước."""

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "user-1"

    # Chưa ghi gì: read trả về profile mặc định, size = 0.
    assert store.file_size(user_id) == 0
    assert "# User Profile" in store.read_text(user_id)

    # write -> read khứ hồi.
    store.write_text(user_id, "# User Profile\n\n- name: Dũng\n")
    assert "name: Dũng" in store.read_text(user_id)
    assert store.file_size(user_id) > 0

    # edit thành công trả về True và thay đúng nội dung.
    assert store.edit_text(user_id, "Dũng", "Thắng") is True
    assert "name: Thắng" in store.read_text(user_id)
    assert "Dũng" not in store.read_text(user_id)

    # edit chuỗi không tồn tại -> False, nội dung giữ nguyên.
    assert store.edit_text(user_id, "không-có-chuỗi-này", "x") is False
    assert "name: Thắng" in store.read_text(user_id)


def test_compact_trigger(tmp_path: Path) -> None:
    """Thread dài phải kích hoạt compaction và giữ lại số message gần nhất."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    thread_id = "long-thread"

    # Nạp nhiều lượt đủ dài để vượt ngưỡng token (mỗi reply thêm 2 message
    # vào compact memory: user + assistant).
    turns = 16
    for i in range(turns):
        agent.reply(
            "user-compact",
            thread_id,
            f"Đây là lượt số {i} với nội dung đủ dài để cộng dồn token cho vượt ngưỡng compact memory.",
        )

    assert agent.compaction_count(thread_id) > 0

    ctx = agent.compact_memory.context(thread_id)
    # Đã có message cũ bị đẩy vào summary -> số message giữ lại nhỏ hơn tổng đã nạp.
    # (Lưu ý: KHÔNG khẳng định <= keep_messages, vì sau lần nén cuối vẫn có thể
    # append thêm message chưa kích hoạt nén lại.)
    assert len(ctx["messages"]) < turns * 2
    assert ctx["summary"].strip() != ""

    # Thread chưa từng dùng thì không có compaction.
    assert agent.compaction_count("thread-không-tồn-tại") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced nhớ fact qua thread mới; Baseline thì không."""

    config = make_config(tmp_path)

    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    user_id = "user-recall"
    fact_msg = "Mình tên là Thắng."
    question = "Tên mình là gì?"

    # Cung cấp fact ở thread A.
    advanced.reply(user_id, "thread-A", fact_msg)
    baseline.reply(user_id, "thread-A", fact_msg)

    # Hỏi lại ở thread B (session mới).
    adv_reply = advanced.reply(user_id, "thread-B", question)["reply"]
    base_reply = baseline.reply(user_id, "thread-B", question)["reply"]

    # Advanced lấy được từ User.md.
    assert "Thắng" in adv_reply
    # Baseline không có long-term memory -> không nhớ qua thread mới.
    assert "Thắng" not in base_reply


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Trên thread dài, prompt context của Advanced bị compact giữ thấp,
    trong khi Baseline phình dần theo lịch sử."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    thread_id = "stress-thread"
    long_msg = (
        "Một lượt hội thoại dài để mô phỏng ngữ cảnh phình to qua nhiều lượt "
        "liên tiếp, thêm chữ cho đủ dài để vượt ngưỡng compact."
    )

    adv_prompt_per_turn: list[int] = []
    base_prompt_per_turn: list[int] = []
    for _ in range(24):
        adv_prompt_per_turn.append(
            advanced.reply("user-stress", thread_id, long_msg)["prompt_tokens"]
        )
        base_prompt_per_turn.append(
            baseline.reply("user-stress", thread_id, long_msg)["prompt_tokens"]
        )

    # Compact phải thực sự chạy trên thread dài này.
    assert advanced.compaction_count(thread_id) > 0

    # Baseline phình dần: lượt cuối tốn prompt hơn lượt đầu.
    assert base_prompt_per_turn[-1] > base_prompt_per_turn[0]

    # Lượt cuối: Advanced (đã nén) tốn prompt context ít hơn Baseline (phình).
    assert adv_prompt_per_turn[-1] < base_prompt_per_turn[-1]

    # Và tổng prompt tokens xử lý qua cả thread của Advanced cũng thấp hơn Baseline.
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
