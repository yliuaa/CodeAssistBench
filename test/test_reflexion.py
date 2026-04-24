"""Tests for Reflexion online evolution helpers."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from cab_evaluation.agents.maintainer_agent import MaintainerAgent
from cab_evaluation.agents.user_agent import UserAgent
from cab_evaluation.core.config import CABConfig
from cab_evaluation.core.models import (
    CommitInfo,
    ConversationMessage,
    GenerationResult,
    IssueData,
    Question,
    SatisfactionStatus,
)
from cab_evaluation.evolution.evo_state import ReflectionUpdatePayload, ReflexionState
from cab_evaluation.workflows.reflexion_workflow import ReflexionWorkflow


@pytest.fixture
def sample_issue_data():
    return IssueData(
        id="1095",
        language="python",
        commit_info=CommitInfo(
            repository="https://github.com/test/repo",
            sha="abc123",
            message="commit message",
            author="tester",
            date="2026-01-01T00:00:00Z",
        ),
        first_question=Question(
            title="Fail to parse formulas",
            body="The parser crashes on formulas.",
            user="alice",
            created_at="2026-01-01T00:00:00Z",
        ),
        comments=[],
        user_satisfaction_condition=["Explain the root cause", "Give a working fix"],
    )


def test_reflexion_state_roundtrip_and_render_budget(tmp_path: Path):
    state = ReflexionState.empty(checkpoint_steps=[0, 2], max_prompt_chars=120)
    state.record_reflection(
        task_id="1",
        question_title="Short title",
        satisfaction_status="NOT_SATISFIED",
        satisfaction_reason="Needs more detail",
        reflection_text="Avoid vague debugging advice and reference the specific failure mode.",
    )
    state.record_reflection(
        task_id="2",
        question_title="Another title",
        satisfaction_status="PARTIALLY_SATISFIED",
        satisfaction_reason="Still missing fix",
        reflection_text="Ground the answer in the user feedback and mention the exact command path.",
    )

    rendered = state.render_for_prompt()
    assert "Reflections:" in rendered
    assert "Task 2" in rendered
    assert len(rendered) <= 120
    assert state.should_checkpoint_after_task()

    checkpoint = tmp_path / "step_002.json"
    state.save(checkpoint)
    loaded = ReflexionState.load(checkpoint)
    assert loaded.step == 2
    assert len(loaded.reflections) == 2
    assert loaded.checkpoint_steps == [0, 2]


def test_maintainer_prompt_includes_evolution_context_only():
    config = CABConfig()
    maintainer = MaintainerAgent(model_name="sonnet37", config=config)
    user = UserAgent(model_name="sonnet37", config=config)

    prompt = maintainer.get_system_prompt(
        repo_url="https://github.com/test/repo",
        commit_hash="abc123",
        evolution_context="You have attempted to answer similar questions before and failed.\nReflections:\n- Task 1: Be more grounded.",
    )
    assert "You have attempted to answer similar questions before and failed." in prompt
    assert "Be more grounded" in prompt

    user_prompt = user.get_system_prompt()
    assert "Past reflections from previous user-feedback tasks" not in user_prompt


def test_reflection_update_payload_from_generation_result(sample_issue_data):
    result = GenerationResult(
        issue_data=sample_issue_data,
        total_conversation_rounds=1,
        original_comment_count=0,
        user_satisfied=False,
        conversation_history=[
            ConversationMessage(role="user", content="original question"),
            ConversationMessage(role="maintainer", content="final answer"),
        ],
        satisfaction_status=SatisfactionStatus.NOT_SATISFIED,
        satisfaction_reason="The answer missed the root cause",
        final_answer="final answer",
        exploration_log="",
    )

    payload = ReflectionUpdatePayload.from_generation_result(result)
    payload_json = json.loads(payload.to_prompt_json())
    assert payload.issue_id == "1095"
    assert payload.satisfaction_status == "NOT_SATISFIED"
    assert payload_json["question_title"] == "Fail to parse formulas"
    assert payload_json["conversation_history"][0]["content"] == "original question"


@pytest.mark.asyncio
async def test_reflexion_workflow_updates_state_and_writes_checkpoint(tmp_path: Path, sample_issue_data):
    config = CABConfig()
    workflow = ReflexionWorkflow(config)

    result = GenerationResult(
        issue_data=sample_issue_data,
        total_conversation_rounds=1,
        original_comment_count=0,
        user_satisfied=False,
        conversation_history=[
            ConversationMessage(role="user", content="original question"),
            ConversationMessage(role="maintainer", content="final answer"),
        ],
        satisfaction_status=SatisfactionStatus.NOT_SATISFIED,
        satisfaction_reason="The answer missed the root cause",
        final_answer="final answer",
        exploration_history=[],
        exploration_log="",
        llm_call_counter={"maintainer": 1, "user": 1},
        prompt_cache={},
    )

    workflow._load_issues = Mock(return_value=([sample_issue_data], [{"number": "1095"}]))
    workflow.generation_workflow.run_generation = AsyncMock(return_value=result)
    workflow._generate_reflection = AsyncMock(return_value="Ground the answer in the exact failure mode.")

    summary = await workflow.run_dataset(
        dataset_file="dataset.jsonl",
        output_dir=str(tmp_path),
        checkpoint_steps=[0, 1],
        agent_model_mapping={"maintainer": "sonnet37", "user": "sonnet37"},
    )

    assert summary["successful"] == 1
    assert (tmp_path / "states" / "step_000.json").exists()
    assert (tmp_path / "states" / "step_001.json").exists()

    result_lines = (tmp_path / "generation_results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(result_lines) == 1
    row = json.loads(result_lines[0])
    assert row["processing_metadata"]["evolution_step_before_task"] == 0
    assert row["processing_metadata"]["evolution_step_after_task"] == 1
