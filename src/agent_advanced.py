from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    ExtractedFact,
    ProfileFact,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    format_profile_markdown,
    parse_profile_markdown,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        # Đồng hồ logic theo từng user (không dùng wall-clock để vẫn tái lập được).
        # Dùng làm mốc tính memory decay cho fact trong User.md.
        self.turn_counter: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Định tuyến giữa offline và live. Bản này dùng đường offline tất định."""

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ----- Quản lý fact trong User.md (confidence + conflict + decay) -----

    def _load_facts(self, user_id: str) -> dict[str, ProfileFact]:
        return parse_profile_markdown(self.profile_store.read_text(user_id))

    def _save_facts(self, user_id: str, facts: dict[str, ProfileFact]) -> None:
        self.profile_store.write_text(user_id, format_profile_markdown(facts))

    def _persist_facts(
        self, user_id: str, updates: dict[str, ExtractedFact], turn: int
    ) -> None:
        """Hợp nhất fact mới vào User.md với 3 bonus:

        - Confidence threshold: bỏ fact dưới ngưỡng (nhiễu, câu đùa).
        - Reinforcement (chống decay): nhắc lại fact cũ -> tăng mentions, làm mới mốc.
        - Conflict handling: đính chính chỉ ghi đè khi fact mới đủ tin cậy so với
          fact cũ (đã trừ decay); KHÔNG giữ đồng thời giá trị cũ sai.
        """
        facts = self._load_facts(user_id)
        changed = False
        for key, ef in updates.items():
            if ef.confidence < self.config.confidence_threshold:
                continue  # confidence threshold
            existing = facts.get(key)
            if existing is None:
                facts[key] = ProfileFact(ef.value, ef.confidence, mentions=1, last_turn=turn)
                changed = True
            elif existing.value == ef.value:
                existing.mentions += 1
                existing.last_turn = turn
                existing.confidence = max(existing.confidence, ef.confidence)
                changed = True
            else:
                old_eff = existing.effective_confidence(turn, self.config.decay_rate)
                if ef.confidence >= old_eff:  # correction thắng fact cũ đã phai
                    facts[key] = ProfileFact(ef.value, ef.confidence, mentions=1, last_turn=turn)
                    changed = True
        if changed:
            self._save_facts(user_id, facts)

    # ----- Đường offline -----

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Đường advanced tất định với đủ 3 lớp memory."""

        # Tăng đồng hồ logic của user (mốc tính decay).
        turn = self.turn_counter.get(user_id, 0) + 1
        self.turn_counter[user_id] = turn

        # 1 + 2. Bóc fact ổn định (kèm confidence) và lưu bền vững vào User.md.
        updates = extract_profile_updates(message)
        if updates:
            self._persist_facts(user_id, updates, turn)

        # 3. Đưa message người dùng vào compact memory (short-term + nén).
        self.compact_memory.append(thread_id, "user", message)

        # 4. Ước lượng prompt context: User.md + summary + message gần nhất.
        #    Nhờ compact nên phần này KHÔNG phình vô hạn ở thread dài.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 5. Sinh câu trả lời dựa trên memory bền vững (trả lời được recall).
        reply = self._offline_response(user_id, thread_id, message)

        # 6. Lưu reply vào compact memory + cộng token agent.
        self.compact_memory.append(thread_id, "assistant", reply)
        agent_tokens = estimate_tokens(reply)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "reply": reply,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        recent = ctx.get("messages", [])

        total = estimate_tokens(profile) + estimate_tokens(summary)
        total += sum(estimate_tokens(m["content"]) for m in recent)
        return total

    # Câu hỏi recall -> field nào trong User.md.
    _FIELD_KEYWORDS = {
        "name": ["tên"],
        "drink": ["đồ uống", "uống", "cà phê"],
        "location": ["ở đâu", "nơi ở", "đang ở", "sống ở"],
        "profession": ["nghề", "làm", "công việc"],
        "style": ["style", "trả lời", "phong cách", "kiểu"],
    }

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Trả lời tất định bằng cách tra fact đã lưu trong User.md."""
        facts = self._load_facts(user_id)
        if not facts:
            return "Mình chưa có thông tin nào về bạn."

        low = message.lower()
        matched: list[tuple[str, str]] = []
        for key, keywords in self._FIELD_KEYWORDS.items():
            if key in facts and any(kw in low for kw in keywords):
                matched.append((key, facts[key].value))

        # Không khớp field nào -> trả về toàn bộ fact, ưu tiên fact "nổi" nhất
        # (tin cậy hiệu dụng cao: tin cậy gốc * reinforcement / decay).
        if not matched:
            turn = self.turn_counter.get(user_id, 0)
            ordered = sorted(
                facts.items(),
                key=lambda kv: kv[1].effective_confidence(turn, self.config.decay_rate),
                reverse=True,
            )
            matched = [(key, fact.value) for key, fact in ordered]

        parts = "; ".join(f"{key}: {value}" for key, value in matched)
        return "Theo những gì mình nhớ -> " + parts + "."

    def _maybe_build_langchain_agent(self):
        """Tùy chọn: dựng live agent với tool đọc/ghi User.md + compact middleware.

        Bản offline đã đủ cho benchmark/test nên phần này để trống an toàn.
        """

        return None
