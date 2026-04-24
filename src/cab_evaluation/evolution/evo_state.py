"""Reflexion evolution state for online CAB runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import GenerationResult


def _truncate_text(text: str, max_chars: int, label: str) -> str:
    """Deterministically truncate long text while preserving recent content."""
    if not text or len(text) <= max_chars:
        return text

    omitted = len(text) - max_chars
    head_chars = max_chars // 3
    tail_chars = max_chars - head_chars
    return (
        f"{text[:head_chars]}\n"
        f"[... omitted {omitted} chars from {label} ...]\n"
        f"{text[-tail_chars:]}"
    )


@dataclass
class ReflectionEntry:
    """One stored Reflexion memory item."""

    task_id: str
    question_title: str
    satisfaction_status: str
    satisfaction_reason: str
    reflection: str

    def render(self) -> str:
        """Render a reflection entry for maintainer prompt injection."""
        return f"- Task {self.task_id} ({self.question_title}): {self.reflection}"


@dataclass
class ReflectionUpdatePayload:
    """Normalized result payload used to generate a new reflection."""

    issue_id: str
    question_title: str
    question_body: str
    final_answer: str
    conversation_history: List[Dict[str, str]]
    satisfaction_status: str
    satisfaction_reason: str
    satisfaction_conditions: List[str]
    exploration_log: str = ""

    @classmethod
    def from_generation_result(cls, result: GenerationResult) -> "ReflectionUpdatePayload":
        """Build a normalized payload from a CAB generation result."""
        return cls(
            issue_id=result.issue_data.id,
            question_title=result.issue_data.first_question.title,
            question_body=result.issue_data.first_question.body,
            final_answer=result.final_answer,
            conversation_history=[
                {"role": message.role, "content": message.content}
                for message in result.conversation_history
            ],
            satisfaction_status=result.satisfaction_status.value,
            satisfaction_reason=result.satisfaction_reason,
            satisfaction_conditions=list(result.issue_data.user_satisfaction_condition),
            exploration_log=result.exploration_log,
        )

    def to_prompt_json(self) -> str:
        """Serialize the payload into prompt-friendly JSON."""
        payload = {
            "issue_id": self.issue_id,
            "question_title": self.question_title,
            "question_body": self.question_body,
            "final_answer": self.final_answer,
            "conversation_history": self.conversation_history,
            "satisfaction_status": self.satisfaction_status,
            "satisfaction_reason": self.satisfaction_reason,
            "satisfaction_conditions": self.satisfaction_conditions,
            "exploration_log": self.exploration_log,
        }
        return json.dumps(payload, indent=2, ensure_ascii=True)


@dataclass
class ReflexionState:
    """Mutable Reflexion memory for online evolution."""

    step: int = 0
    reflections: List[ReflectionEntry] = field(default_factory=list)
    checkpoint_steps: List[int] = field(default_factory=list)
    max_prompt_chars: int = field(
        default_factory=lambda: int(os.getenv("CAB_REFLEXION_MEMORY_CHARS", "4000"))
    )

    @property
    def method_name(self) -> str:
        """Name of the evolution method."""
        return "reflexion"

    @property
    def reflection_header(self) -> str:
        """Official-style Reflexion header used when injecting memory."""
        return (
            "You have attempted to answer similar questions before and failed. "
            "The following reflection(s) give a plan to avoid failing in the same way. "
            "Use them to improve your strategy for the current question.\n"
        )

    @classmethod
    def empty(
        cls,
        checkpoint_steps: Optional[List[int]] = None,
        max_prompt_chars: Optional[int] = None,
    ) -> "ReflexionState":
        """Create an empty state."""
        return cls(
            step=0,
            reflections=[],
            checkpoint_steps=sorted(set(checkpoint_steps or [])),
            max_prompt_chars=max_prompt_chars
            if max_prompt_chars is not None
            else int(os.getenv("CAB_REFLEXION_MEMORY_CHARS", "4000")),
        )

    def render_for_prompt(self) -> str:
        """Render bounded reflection memory for maintainer prompt injection."""
        if not self.reflections:
            return ""

        selected: List[str] = []
        current_length = 0
        for entry in reversed(self.reflections):
            rendered = entry.render()
            addition = rendered if not selected else f"{rendered}\n" + "\n".join(reversed(selected))
            if len(addition) > self.max_prompt_chars and selected:
                break
            selected.append(rendered)
            current_length += len(rendered) + 1
            if current_length >= self.max_prompt_chars:
                break

        rendered_memory = self.reflection_header + "Reflections:\n- " + "\n- ".join(reversed(selected))
        return _truncate_text(rendered_memory, self.max_prompt_chars, "reflexion memory")

    def record_reflection(
        self,
        task_id: str,
        question_title: str,
        satisfaction_status: str,
        satisfaction_reason: str,
        reflection_text: str,
    ) -> None:
        """Append a reflection and advance the evolution step."""
        self.reflections.append(
            ReflectionEntry(
                task_id=task_id,
                question_title=question_title,
                satisfaction_status=satisfaction_status,
                satisfaction_reason=satisfaction_reason,
                reflection=reflection_text.strip(),
            )
        )
        self.step += 1

    def should_checkpoint_after_task(self) -> bool:
        """Return whether the current step is a checkpoint."""
        return self.step in set(self.checkpoint_steps)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full state for checkpointing."""
        return {
            "method": self.method_name,
            "step": self.step,
            "checkpoint_steps": list(self.checkpoint_steps),
            "max_prompt_chars": self.max_prompt_chars,
            "source_task_ids": [entry.task_id for entry in self.reflections],
            "reflections": [asdict(entry) for entry in self.reflections],
            "rendered_memory_snapshot": self.render_for_prompt(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReflexionState":
        """Reconstruct a state from serialized data."""
        reflections = [ReflectionEntry(**entry) for entry in payload.get("reflections", [])]
        return cls(
            step=payload.get("step", 0),
            reflections=reflections,
            checkpoint_steps=list(payload.get("checkpoint_steps", [])),
            max_prompt_chars=payload.get(
                "max_prompt_chars", int(os.getenv("CAB_REFLEXION_MEMORY_CHARS", "4000"))
            ),
        )

    def save(self, path: Path) -> None:
        """Save the state as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReflexionState":
        """Load the state from JSON."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
