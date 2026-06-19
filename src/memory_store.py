from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """
    text = text.strip()
    if not text:
        return 0
    return max(1,len(text)//4)

@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # TODO: slugify or sanitize the user id before building the file path.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id).strip("_")
        return self.root_dir / f"{safe or 'user'}.md"

    def read_text(self, user_id: str) -> str:
        # TODO: return file content or an empty default markdown profile.
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile\n\n"   # default markdown khi chưa có file

    def write_text(self, user_id: str, content: str) -> Path:
        # TODO: write markdown to disk and return the file path.
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)   # state/profiles/ có thể chưa tồn tại
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        # TODO: replace one occurrence inside User.md and return whether it changed.
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        # TODO: return the current file size in bytes.
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = message.strip()
    low = text.lower()

    # 1. Bỏ qua câu HỎI: không trích fact từ câu người dùng đang hỏi lại.
    question_markers = (
        "?", "là gì", "ở đâu", "thế nào", "bao nhiêu",
        "khi nào", "nhắc lại", "nhớ lại", "có thể nhắc",
    )
    if any(marker in low for marker in question_markers):
        return {}

    facts: dict[str, str] = {}

    # 2. name: "mình tên là DũngCT"
    m = re.search(r"tên(?:\s+mình)?\s+là\s+([A-Za-zÀ-ỹ0-9]+)", text, re.I)
    if m:
        facts["name"] = m.group(1).strip(" .")

    # 3. location: yêu cầu có chủ ngữ trước "ở" (mình/tôi/đang/giờ) để tránh
    #    bắt nhầm "nơi ở đã thay đổi". Lưu ý: dải À-Ỹ gồm cả chữ thường có dấu,
    #    nên không thể dựa vào "chữ Hoa" để lọc — phải neo bằng ngữ cảnh.
    m = re.search(
        r"(?:mình|tôi|giờ|hiện(?: tại)?|đang)\s+(?:đang\s+)?ở\s+([A-Za-zÀ-ỹ]+(?:\s+[A-Za-zÀ-ỹ]+)?)",
        text, re.I,
    )
    if m:
        facts["location"] = m.group(1).strip(" .")

    # 4. profession: "đang làm backend engineer cho startup AI"
    m = re.search(r"làm\s+([A-Za-zÀ-ỹ ]+?)\s+(?:cho|tại|ở)\b", text, re.I)
    if m:
        facts["profession"] = m.group(1).strip(" .")

    # 5. drink: "đồ uống yêu thích là cà phê sữa đá" / "uống cà phê sữa đá như cũ"
    m = re.search(r"(?:đồ uống|thức uống)[^.]*?là\s+([A-Za-zÀ-ỹ ]+)", text, re.I)
    if not m:
        m = re.search(r"\buống\s+([A-Za-zÀ-ỹ ]+?)(?:\s+như cũ|\.|$)", text, re.I)
    if m:
        facts["drink"] = m.group(1).strip(" .")

    # 6. style: "trả lời ngắn gọn, rõ ý và có ví dụ thực tế".
    #    Bỏ qua capture quá ngắn hoặc chỉ là đại từ (vd "mình thích", "gọn").
    m = re.search(r"trả lời\s+([A-Za-zÀ-ỹ ,]+)", text, re.I)
    if m:
        style = m.group(1).strip(" .,")
        if len(style) >= 6 and not style.lower().startswith(("mình", "tôi", "bạn", "thành")):
            facts["style"] = style

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """
    if not messages:
        return ""
    lines = [f"{m['role']}: {m['content']}" for m in messages[-max_items:]]
    return "Tóm tắt hội thoại trước:\n" + "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _new_state(self) -> dict[str, object]:
        # State mặc định cho một thread mới.
        return {"messages": [], "summary": "", "compactions": 0}

    def append(self, thread_id: str, role: str, content: str) -> None:
        # 1. Tạo state nếu thread chưa tồn tại.
        st = self.state.setdefault(thread_id, self._new_state())
        # 2. Thêm message mới vào cuối.
        st["messages"].append({"role": role, "content": content})
        # 3. Nén nếu vượt ngưỡng.
        self._maybe_compact(st)

    def _maybe_compact(self, st: dict[str, object]) -> None:
        messages = st["messages"]
        # Tổng token = summary đã có + toàn bộ message đang giữ.
        total = estimate_tokens(st["summary"]) + sum(
            estimate_tokens(m["content"]) for m in messages
        )
        # Chỉ nén khi vượt ngưỡng VÀ còn nhiều hơn số message cần giữ lại.
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        # Phần cũ đem nén, phần gần nhất giữ nguyên văn.
        old = messages[: -self.keep_messages]
        st["messages"] = messages[-self.keep_messages :]
        new_summary = summarize_messages(old)
        st["summary"] = (st["summary"] + "\n" + new_summary).strip()
        st["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        # Trả state của thread, hoặc state rỗng nếu chưa có (không tự tạo).
        return self.state.get(thread_id, self._new_state())

    def compaction_count(self, thread_id: str) -> int:
        return self.state.get(thread_id, {}).get("compactions", 0)
