from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Trả về phản hồi + hạch toán token.

        Baseline chỉ có một đường offline tất định. `user_id` bị bỏ qua có chủ ý:
        baseline KHÔNG dùng hồ sơ người dùng, chỉ nhớ trong phạm vi `thread_id`.
        """

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        # Tổng token agent đã sinh trong thread này (0 nếu thread chưa tồn tại).
        sess = self.sessions.get(thread_id)
        return sess.token_usage if sess else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        # Lượng ngữ cảnh prompt baseline phải xử lý dồn lại qua các lượt.
        sess = self.sessions.get(thread_id)
        return sess.prompt_tokens_processed if sess else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Hành vi offline đơn giản, "ngây thơ" có chủ ý.

        - Lưu message vào session của ĐÚNG thread đó.
        - prompt context = toàn bộ history thread -> phình dần theo độ dài.
        - Reply tất định, KHÔNG truy xuất fact của thread/khác session.
        """

        sess = self.sessions.setdefault(thread_id, SessionState())
        sess.messages.append({"role": "user", "content": message})

        # Baseline kéo lại TOÀN BỘ lịch sử thread mỗi lượt -> chi phí prompt tăng dần.
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in sess.messages)
        sess.prompt_tokens_processed += prompt_tokens

        reply = f"Mình đã ghi nhận: {message}"
        sess.messages.append({"role": "assistant", "content": reply})

        agent_tokens = estimate_tokens(reply)
        sess.token_usage += agent_tokens

        return {
            "reply": reply,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Tùy chọn: dựng agent LangChain thật khi đã cài dependency.

        Dùng `build_chat_model(self.config.model)` để chạy với provider bất kỳ.
        Bản offline đã đủ cho benchmark/test nên phần này để trống an toàn.
        """

        return None
