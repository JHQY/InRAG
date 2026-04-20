# prompt_template.py
# -*- coding: utf-8 -*-
from typing import List


# ---------- RAFT 专用模板 ----------
RAFT_SYSTEM = (
    "你是一个保险助手，请严格根据提供的检索结果回答用户问题，"
    "不得使用检索结果以外的任何知识。"
    "如果检索结果不包含回答所需信息，请明确告知用户无法作答。"
)


def build_prompt_raft(query: str, ref_chunks: List[dict]) -> tuple:
    """
    为 RAFT 模型构建 (system_prompt, user_message)。

    Args:
        query: 用户问题
        ref_chunks: 检索结果列表，每项包含 'contract_name'（str）和 'text'（str）

    Returns:
        (system_prompt, user_message) — 对应 messages 中 system/user 两条
    """
    parts = []
    for i, chunk in enumerate(ref_chunks, 1):
        text = chunk['text'][:400]  # 截短每个chunk避免VRAM峰值超限
        parts.append(
            f"检索结果{i}:\n"
            f"合同名称：{chunk['contract_name']}\n"
            f"条款内容：{text}"
        )
    parts.append(f"问题：{query}")
    user_message = "\n\n".join(parts)
    return RAFT_SYSTEM, user_message