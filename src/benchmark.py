from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Đọc danh sách hội thoại từ file JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """0 nếu không nhớ fact nào, 1 nếu nhớ đủ, 0.5 nếu nhớ một phần."""

    if not expected:
        return 0.0
    low = answer.lower()
    hits = sum(1 for fact in expected if fact.lower() in low)
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Chất lượng nhẹ cho offline: rỗng -> 0, có ít nhất 1 fact đúng -> 1, còn lại 0.5."""

    if not answer.strip():
        return 0.0
    low = answer.lower()
    if any(fact.lower() in low for fact in expected):
        return 1.0
    return 0.5


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Đánh giá một agent trên nhiều hội thoại.

    Quan trọng: câu hỏi recall được hỏi ở MỘT THREAD MỚI để kiểm tra khả năng
    nhớ xuyên session (baseline sẽ trượt, advanced dựa vào User.md).
    """

    agent_tokens = 0
    prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    compactions = 0
    user_ids: set[str] = set()

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]
        user_ids.add(user_id)

        # 1. Nạp toàn bộ lượt hội thoại vào agent.
        for turn in conv["turns"]:
            result = agent.reply(user_id, thread_id, turn)
            agent_tokens += result["agent_tokens"]
            prompt_tokens += result["prompt_tokens"]
        compactions += agent.compaction_count(thread_id)

        # 2. Hỏi recall ở THREAD MỚI (xuyên session).
        recall_thread = f"{thread_id}-recall"
        for question in conv["recall_questions"]:
            result = agent.reply(user_id, recall_thread, question["question"])
            agent_tokens += result["agent_tokens"]
            prompt_tokens += result["prompt_tokens"]
            expected = question["expected_contains"]
            recall_scores.append(recall_points(result["reply"], expected))
            quality_scores.append(heuristic_quality(result["reply"], expected))
        compactions += agent.compaction_count(recall_thread)

    # 3. Memory growth: baseline không có User.md -> 0.
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(uid) for uid in user_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=_average(recall_scores),
        response_quality=_average(quality_scores),
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """In bảng so sánh. Dùng tabulate nếu có, không thì fallback thủ công."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]

    try:
        from tabulate import tabulate

        return tabulate(table, headers=headers, tablefmt="github")
    except ImportError:
        lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
        lines += [" | ".join(str(cell) for cell in row) for row in table]
        return "\n".join(lines)


def _run_suite(title: str, dataset_path: Path, config) -> None:
    conversations = load_conversations(dataset_path)
    # Agent mới mỗi suite để state không rò rỉ giữa các benchmark.
    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config=config, force_offline=True), conversations, config),
        run_agent_benchmark("Advanced", AdvancedAgent(config=config, force_offline=True), conversations, config),
    ]
    print(f"\n## {title}\n")
    print(format_rows(rows))


def main() -> None:
    """Chạy cả hai suite: standard và long-context stress."""

    config = load_config(Path(__file__).resolve().parent.parent)

    _run_suite("Standard Benchmark", config.data_dir / "conversations.json", config)
    _run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config)


if __name__ == "__main__":
    main()
