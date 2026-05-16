from __future__ import annotations

from .data import Task


SYSTEM = (
    "You are a careful mathematical debater. Your goal is truth, not agreement. "
    "Use concise reasoning. Return exactly one JSON object and no Markdown. "
    'The JSON must include "answer" as an integer and "confidence" as a number from 0 to 1.'
)


def private_answer_messages(task: Task, agent_id: int | None = None) -> list[dict[str, str]]:
    agent = f"Agent {agent_id}" if agent_id is not None else "Agent"
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"{agent}, independently solve the task before seeing any peer answers. "
                'Return exactly {"answer": <integer>, "confidence": <0-1>, "rationale": "<short check>"}.\n\n'
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
                'Give your updated solution as exactly one JSON object: '
                '{"answer": <integer>, "confidence": <0-1>, "rationale": "<short check>"}'
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
                "Respond as exactly one JSON object with these keys: "
                '"private_answer", "majority_risk", "error_check", "minority_report", "answer", "confidence". '
                '"answer" must be an integer and "confidence" must be a number from 0 to 1.'
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
                "Now produce exactly one JSON object with keys "
                '"private_answer", "majority_risk", "error_check", "minority_report", "answer", "confidence". '
                '"answer" must be an integer and "confidence" must be a number from 0 to 1.'
            ),
        },
    ]


def _format_peers(peer_messages: list[str]) -> str:
    if not peer_messages:
        return "No peer messages."
    return "\n\n".join(f"Peer {idx + 1}:\n{msg}" for idx, msg in enumerate(peer_messages))


def json_private_answer(answer: str | int, confidence: float = 0.95, rationale: str = "computed with standard order of operations") -> str:
    return (
        '{"answer": '
        f"{int(answer)}, "
        f'"confidence": {confidence:.2f}, '
        f'"rationale": "{rationale}"'
        "}"
    )


def json_anti_conformity_answer(
    answer: str | int,
    private_answer: str | int,
    wrong_majority_answer: str | int,
    confidence: float = 0.95,
) -> str:
    return (
        '{"private_answer": '
        f"{int(private_answer)}, "
        '"majority_risk": "the peer majority may be using a precedence or arithmetic shortcut", '
        '"error_check": "recompute multiplication before addition and subtraction", '
        f'"minority_report": "wrong majority answer was {int(wrong_majority_answer)}", '
        f'"answer": {int(answer)}, '
        f'"confidence": {confidence:.2f}'
        "}"
    )
