from __future__ import annotations

from .data import Task


SYSTEM = (
    "You are a careful mathematical debater. Your goal is truth, not agreement. "
    "Use concise reasoning. Always end with ANSWER: <integer> and CONFIDENCE: <0-100>."
)


def private_answer_messages(task: Task, agent_id: int | None = None) -> list[dict[str, str]]:
    agent = f"Agent {agent_id}" if agent_id is not None else "Agent"
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"{agent}, independently solve the task before seeing any peer answers. "
                "State the key operation-order assumption you used.\n\n"
                f"{task.question}"
            ),
        },
    ]


def vanilla_update_messages(task: Task, own_previous: str, peer_messages: list[str], round_idx: int, agent_id: int) -> list[dict[str, str]]:
    peers = _format_peers(peer_messages)
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Agent {agent_id}, this is debate round {round_idx}. "
                "You may revise your answer after reading the other agents.\n\n"
                f"Task:\n{task.question}\n\n"
                f"Your previous answer:\n{own_previous}\n\n"
                f"Other agents:\n{peers}\n\n"
                "Give your updated solution. End with ANSWER and CONFIDENCE."
            ),
        },
    ]


def anti_conformity_update_messages(
    task: Task,
    own_commitment: str,
    peer_messages: list[str],
    round_idx: int,
    agent_id: int,
) -> list[dict[str, str]]:
    peers = _format_peers(peer_messages)
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Agent {agent_id}, this is anti-conformity debate round {round_idx}. "
                "Your private commitment is preserved below. Do not switch merely because a majority disagrees. "
                "You may switch only if you can locate a concrete error in your earlier reasoning.\n\n"
                f"Task:\n{task.question}\n\n"
                f"Your private commitment:\n{own_commitment}\n\n"
                f"Peer commitments and arguments:\n{peers}\n\n"
                "Respond with these fields:\n"
                "1. PRIVATE_ANSWER: your original answer.\n"
                "2. MAJORITY_RISK: one sentence on how the majority could be wrong.\n"
                "3. ERROR_CHECK: the exact arithmetic/order-of-operations check.\n"
                "4. MINORITY_REPORT: the best non-majority answer if any.\n"
                "5. ANSWER: <integer>\n"
                "6. CONFIDENCE: <0-100>"
            ),
        },
    ]


def rl_final_messages(task: Task, own_commitment: str, peer_messages: list[str]) -> list[dict[str, str]]:
    peers = _format_peers(peer_messages)
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "You are being trained as the final debater in a truth-seeking protocol. "
                "You receive your private answer and peer answers. Your reward depends on correctness, "
                "not on matching peers. Preserve a correct private answer even if peers form a wrong majority.\n\n"
                f"Task:\n{task.question}\n\n"
                f"Your private commitment:\n{own_commitment}\n\n"
                f"Peer answers:\n{peers}\n\n"
                "Now produce the final anti-conformity response with MAJORITY_RISK, ERROR_CHECK, "
                "ANSWER, and CONFIDENCE."
            ),
        },
    ]


def _format_peers(peer_messages: list[str]) -> str:
    if not peer_messages:
        return "No peer messages."
    return "\n\n".join(f"Peer {idx + 1}:\n{msg}" for idx, msg in enumerate(peer_messages))
