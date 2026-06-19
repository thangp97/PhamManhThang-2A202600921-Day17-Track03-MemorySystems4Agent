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


@dataclass
class ExtractedFact:
    """Một fact vừa bóc ra từ message, kèm độ tin cậy [0..1].

    `confidence` là đầu vào cho bonus *confidence threshold* và *conflict
    handling*: clause có dấu hiệu phủ định / đùa cợt sẽ bị hạ tin cậy mạnh nên
    không được ghi vào `User.md`.
    """

    value: str
    confidence: float


# Dấu hiệu cho thấy clause KHÔNG khẳng định một fact ổn định (đùa, nhiễu, đính
# chính cái cũ). Có bất kỳ marker nào -> hạ mạnh độ tin cậy của fact trong clause.
_NEGATION_MARKERS = (
    "không phải", "chỉ là", "đùa", "lúc đầu", "trước đó", "đừng",
    "vừa bay", "gây nhiễu", "hay là chuyển", "tưởng",
)
# Dấu hiệu fact là trạng thái HIỆN TẠI / ổn định -> tăng nhẹ độ tin cậy.
_AFFIRM_MARKERS = (
    "hiện tại", "hiện ", "đang ", "từ tuần này", "nơi ở hiện tại",
    "vẫn là", "vẫn giữ", "thực ra",
)
_QUESTION_MARKERS = (
    "?", "là gì", "ở đâu", "thế nào", "bao nhiêu",
    "khi nào", "nhắc lại", "nhớ lại", "có thể nhắc", "đâu mới",
)
# Độ tin cậy nền theo từng field (pattern càng rõ -> nền càng cao).
_BASE_CONFIDENCE = {
    "name": 0.9,
    "location": 0.8,
    "profession": 0.8,
    "drink": 0.75,
    "style": 0.7,
}


def _split_clauses(message: str) -> list[str]:
    """Tách message thành các clause nhỏ để chấm tin cậy theo ngữ cảnh cục bộ.

    Cắt theo câu (. ! ? xuống dòng) rồi theo dấu phẩy và liên từ "nhưng" — nhờ
    vậy "Lúc đầu mình nói ở Huế, nhưng thực ra ... ở Đà Nẵng" tách được phần
    đính chính (Huế) khỏi phần hiện tại (Đà Nẵng).
    """
    parts = re.split(r"[.!?\n]|(?:\bnhưng\b)|,", message)
    return [p.strip() for p in parts if p.strip()]


def _score(field: str, clause_low: str) -> float:
    conf = _BASE_CONFIDENCE[field]
    if any(mark in clause_low for mark in _NEGATION_MARKERS):
        conf *= 0.25  # nhiễu / đùa / đính chính cũ -> rơi xuống dưới ngưỡng
    elif any(mark in clause_low for mark in _AFFIRM_MARKERS):
        conf = min(0.99, conf + 0.1)
    return conf


def extract_profile_updates(message: str) -> dict[str, ExtractedFact]:
    """Bóc các fact ổn định kèm độ tin cậy.

    Quy trình:
    1. Bỏ qua câu hỏi (không lưu fact từ câu người dùng đang hỏi lại).
    2. Tách clause, chạy regex từng field trên mỗi clause.
    3. Chấm tin cậy theo marker phủ định/khẳng định trong clause.
    4. Mỗi field giữ ứng viên có tin cậy cao nhất trong message.
    """

    if any(marker in message.lower() for marker in _QUESTION_MARKERS):
        return {}

    best: dict[str, ExtractedFact] = {}

    def offer(field: str, value: str, clause_low: str) -> None:
        value = value.strip(" .,:")
        if not value:
            return
        conf = _score(field, clause_low)
        cur = best.get(field)
        if cur is None or conf > cur.confidence:
            best[field] = ExtractedFact(value=value, confidence=conf)

    for clause in _split_clauses(message):
        low = clause.lower()

        # name: "mình tên là DũngCT Stress" (cho phép 1-3 token để bắt tên ghép)
        m = re.search(
            r"tên(?:\s+mình)?\s+là\s+([A-Za-zÀ-ỹ0-9]+(?:\s+[A-Za-zÀ-ỹ0-9]+){0,2})",
            clause, re.I,
        )
        if m:
            offer("name", m.group(1), low)

        # location: cần chủ ngữ/ngữ cảnh trước "ở" để tránh bắt nhầm.
        m = re.search(
            r"(?:mình|tôi|giờ|hiện(?: tại)?|đang|việc)\s+(?:đang\s+)?(?:làm việc\s+)?ở\s+"
            r"([A-Za-zÀ-ỹ]+(?:\s+[A-Za-zÀ-ỹ]+)?)",
            clause, re.I,
        )
        if m:
            offer("location", m.group(1), low)

        # profession: "đang làm MLOps engineer cho ..." hoặc "nghề ... là MLOps engineer"
        m = re.search(r"làm\s+([A-Za-zÀ-ỹ ]+?)\s+(?:cho|tại|ở)\b", clause, re.I)
        if m:
            offer("profession", m.group(1), low)
        m = re.search(r"nghề(?:\s+nghiệp)?[^:]*?\blà\s+([A-Za-zÀ-ỹ ]+)", clause, re.I)
        if m:
            offer("profession", m.group(1), low)

        # drink
        m = re.search(r"(?:đồ uống|thức uống)[^.]*?là\s+([A-Za-zÀ-ỹ ]+)", clause, re.I)
        if not m:
            m = re.search(r"\buống\s+([A-Za-zÀ-ỹ ]+?)(?:\s+như cũ|$)", clause, re.I)
        if m:
            offer("drink", m.group(1), low)

        # style: "trả lời ngắn gọn thành 3 bullet" (cho phép cả chữ số).
        m = re.search(r"trả lời\s+([A-Za-zÀ-ỹ0-9 ]+)", clause, re.I)
        if m:
            style = m.group(1).strip(" .,")
            if len(style) >= 6 and not style.lower().startswith(
                ("mình", "tôi", "bạn", "thành", "cũng", "vẫn", "như", "đã")
            ):
                offer("style", style, low)

    return best


@dataclass
class ProfileFact:
    """Fact bền vững trong `User.md` kèm metadata cho bonus decay/conflict.

    - `confidence`: độ tin cậy lúc ghi.
    - `mentions`: số lần fact này được nhắc lại (reinforcement chống decay).
    - `last_turn`: lượt gần nhất fact được xác nhận (mốc tính decay).
    """

    value: str
    confidence: float = 0.0
    mentions: int = 1
    last_turn: int = 0

    def effective_confidence(self, current_turn: int, decay_rate: float) -> float:
        """Tin cậy hiệu dụng: giảm theo số lượt im lặng, tăng theo số lần nhắc lại."""
        idle = max(0, current_turn - self.last_turn)
        reinforcement = 1.0 + 0.15 * (self.mentions - 1)
        return self.confidence * (decay_rate ** idle) * reinforcement


# Dòng fact trong User.md: "- key: value  <!-- conf=0.90 mentions=2 turn=5 -->".
# Metadata để trong HTML comment nên file vẫn đọc được như markdown thường.
_FACT_LINE = re.compile(
    r"-\s*([A-Za-z_]+):\s*(.+?)"
    r"(?:\s*<!--\s*conf=([\d.]+)\s+mentions=(\d+)\s+turn=(\d+)\s*-->)?\s*$"
)


def parse_profile_markdown(text: str) -> dict[str, ProfileFact]:
    """Đọc `User.md` thành dict field -> ProfileFact (tương thích cả dòng cũ không metadata)."""
    facts: dict[str, ProfileFact] = {}
    for line in text.splitlines():
        m = _FACT_LINE.match(line)
        if not m:
            continue
        key, value, conf, mentions, turn = m.groups()
        if conf is None:
            facts[key] = ProfileFact(value=value.strip())
        else:
            facts[key] = ProfileFact(
                value=value.strip(),
                confidence=float(conf),
                mentions=int(mentions),
                last_turn=int(turn),
            )
    return facts


def format_profile_markdown(facts: dict[str, ProfileFact]) -> str:
    """Ghi dict ProfileFact ra markdown `User.md`."""
    lines = ["# User Profile", ""]
    for key, fact in facts.items():
        lines.append(
            f"- {key}: {fact.value}  "
            f"<!-- conf={fact.confidence:.2f} mentions={fact.mentions} turn={fact.last_turn} -->"
        )
    return "\n".join(lines) + "\n"


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
