"""Online Reflexion evolution workflow for CAB."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..agents.llm_service import LLMService
from ..core.config import CABConfig
from ..core.models import IssueData
from ..evolution.evo_state import ReflectionUpdatePayload, ReflexionState
from ..utils.data_processor import DataProcessor
from .generation_workflow import GenerationWorkflow

logger = logging.getLogger(__name__)


class ReflexionWorkflow:
    """Runs CAB generation while updating a Reflexion memory online."""

    def __init__(self, config: Optional[CABConfig] = None):
        self.config = config or CABConfig()
        self.data_processor = DataProcessor()
        self.generation_workflow = GenerationWorkflow(self.config)
        self.llm_service = LLMService(self.config)

    async def run_dataset(
        self,
        dataset_file: str,
        output_dir: str,
        agent_model_mapping: Optional[Dict[str, str]] = None,
        agent_framework_mapping: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        checkpoint_steps: Optional[List[int]] = None,
        max_prompt_chars: Optional[int] = None,
        enable_ast_tools: bool = True,
    ) -> Dict[str, int]:
        """Run online Reflexion evolution on a JSONL dataset."""
        issues, raw_data = self._load_issues(dataset_file, language)

        run_dir = Path(output_dir)
        states_dir = run_dir / "states"
        results_file = run_dir / "generation_results.jsonl"
        metadata_file = run_dir / "metadata.json"

        run_dir.mkdir(parents=True, exist_ok=True)

        state = ReflexionState.empty(
            checkpoint_steps=checkpoint_steps or [0, 5, 10, 20, 50],
            max_prompt_chars=max_prompt_chars,
        )
        self._write_metadata(
            metadata_file,
            dataset_file,
            language,
            agent_model_mapping,
            agent_framework_mapping,
            state,
        )

        if 0 in state.checkpoint_steps:
            state.save(states_dir / "step_000.json")

        successful_count = 0
        failed_count = 0

        with results_file.open("w", encoding="utf-8") as output_handle:
            for issue_data in issues:
                reflection_context = state.render_for_prompt() or None
                step_before_task = state.step
                try:
                    result = await self.generation_workflow.run_generation(
                        issue_data,
                        agent_model_mapping=agent_model_mapping,
                        agent_framework_mapping=agent_framework_mapping,
                        enable_ast_tools=enable_ast_tools,
                        maintainer_evolution_context=reflection_context,
                    )

                    reflection_text = await self._generate_reflection(
                        result,
                        agent_model_mapping.get("maintainer") if agent_model_mapping else None,
                    )
                    state.record_reflection(
                        task_id=result.issue_data.id,
                        question_title=result.issue_data.first_question.title,
                        satisfaction_status=result.satisfaction_status.value,
                        satisfaction_reason=result.satisfaction_reason,
                        reflection_text=reflection_text,
                    )

                    checkpoint_path = None
                    if state.should_checkpoint_after_task():
                        checkpoint_path = states_dir / f"step_{state.step:03d}.json"
                        state.save(checkpoint_path)

                    result_dict = self._generation_result_to_dict(
                        result,
                        raw_data=raw_data,
                        dataset_file=dataset_file,
                        language=language,
                        agent_model_mapping=agent_model_mapping or {},
                        agent_framework_mapping=agent_framework_mapping or {},
                        step_before_task=step_before_task,
                        state=state,
                        checkpoint_path=checkpoint_path,
                    )
                    output_handle.write(json.dumps(result_dict, default=str) + "\n")
                    output_handle.flush()
                    successful_count += 1
                    logger.info(
                        "Reflexion step %s complete for issue %s",
                        state.step,
                        issue_data.id,
                    )
                except Exception as exc:
                    failed_count += 1
                    logger.error("Error processing issue %s: %s", issue_data.id, exc)
                    error_result = {
                        "issue_id": issue_data.id,
                        "question_title": issue_data.first_question.title,
                        "question_body": issue_data.first_question.body,
                        "error": str(exc),
                        "processing_metadata": {
                            "workflow": "generation_reflexion_online",
                            "timestamp": datetime.now().isoformat(),
                            "error_occurred": True,
                            "evolution_method": "reflexion",
                            "evolution_step_before_task": step_before_task,
                            "input_file": dataset_file,
                        },
                    }
                    output_handle.write(json.dumps(error_result, default=str) + "\n")
                    output_handle.flush()

        return {
            "successful": successful_count,
            "failed": failed_count,
            "total": len(issues),
        }

    def _load_issues(
        self,
        dataset_file: str,
        language: Optional[str],
    ) -> tuple[List[IssueData], List[dict]]:
        """Load and optionally filter issues from dataset."""
        raw_data = self.data_processor.load_jsonl_data([dataset_file])
        if language:
            raw_data = [item for item in raw_data if item.get("language", "").lower() == language.lower()]

        issues: List[IssueData] = []
        for item in raw_data:
            issues.append(self.data_processor.load_issue_data_from_dict(item))
        logger.info("Loaded %s issues for Reflexion evolution from %s", len(issues), dataset_file)
        return issues, raw_data

    def _write_metadata(
        self,
        metadata_file: Path,
        dataset_file: str,
        language: Optional[str],
        agent_model_mapping: Optional[Dict[str, str]],
        agent_framework_mapping: Optional[Dict[str, str]],
        state: ReflexionState,
    ) -> None:
        """Write run metadata once per evolution run."""
        payload = {
            "workflow": "generation_reflexion_online",
            "timestamp": datetime.now().isoformat(),
            "dataset_file": dataset_file,
            "language_filter": language,
            "agent_model_mapping": agent_model_mapping or {},
            "agent_framework_mapping": agent_framework_mapping or {},
            "evolution_method": state.method_name,
            "checkpoint_steps": list(state.checkpoint_steps),
            "max_prompt_chars": state.max_prompt_chars,
        }
        metadata_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async def _generate_reflection(
        self,
        result,
        maintainer_model_name: Optional[str],
    ) -> str:
        """Generate one Reflexion update from a completed generation result."""
        payload = ReflectionUpdatePayload.from_generation_result(result)
        model_name = maintainer_model_name or self.config.default_maintainer_model
        model_config = self.config.get_model_config(model_name)

        system_prompt = (
            "You are an advanced reasoning agent that can improve based on self reflection. "
            "You will be given a previous trial in which a software maintainer answered a user question "
            "and received user feedback indicating the answer was not fully satisfactory. "
            "In a few sentences, diagnose a possible reason for failure and devise a new, concise, "
            "high-level plan that aims to mitigate the same failure in future tasks. "
            "Use complete sentences. Return only the reflection text."
        )
        user_prompt = (
            "Previous trial:\n"
            f"{payload.to_prompt_json()}\n\n"
            "Reflection:"
        )
        reflection = await self.llm_service.call_model(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_config=model_config,
            agent_type="reflexion_updater",
            issue_id=payload.issue_id,
            max_retries=3,
        )
        return reflection.strip().strip('"')

    def _generation_result_to_dict(
        self,
        result,
        raw_data: List[dict],
        dataset_file: str,
        language: Optional[str],
        agent_model_mapping: Dict[str, str],
        agent_framework_mapping: Dict[str, str],
        step_before_task: int,
        state: ReflexionState,
        checkpoint_path: Optional[Path],
    ) -> dict:
        """Serialize a generation result with evolution metadata."""
        original_issue = None
        for orig_item in raw_data:
            if str(orig_item.get("number")) == str(result.issue_data.id):
                original_issue = orig_item
                break

        framework_mapping = {
            "maintainer": agent_framework_mapping.get("maintainer", "strands"),
            "user": agent_framework_mapping.get("user", "strands"),
            "judge": agent_framework_mapping.get("judge", "strands"),
        }

        return {
            "issue_id": result.issue_data.id,
            "question_title": result.issue_data.first_question.title,
            "question_body": result.issue_data.first_question.body,
            "user": result.issue_data.first_question.user,
            "language": result.issue_data.language,
            "repository": result.issue_data.commit_info.repository,
            "commit_sha": result.issue_data.commit_info.sha,
            "final_answer": result.final_answer,
            "user_satisfied": result.user_satisfied,
            "satisfaction_status": result.satisfaction_status.value,
            "satisfaction_reason": result.satisfaction_reason,
            "total_conversation_rounds": result.total_conversation_rounds,
            "original_comment_count": result.original_comment_count,
            "conversation_history": [
                {"role": msg.role, "content": msg.content}
                for msg in result.conversation_history
            ],
            "exploration_history": result.exploration_history,
            "exploration_log": result.exploration_log,
            "llm_call_counter": result.llm_call_counter,
            "prompt_cache": result.prompt_cache,
            "original_metadata": original_issue if original_issue else {},
            "processing_metadata": {
                "workflow": "generation_reflexion_online",
                "timestamp": datetime.now().isoformat(),
                "agent_model_mapping": agent_model_mapping,
                "agent_framework_mapping": framework_mapping,
                "input_file": dataset_file,
                "language_filter": language,
                "evolution_method": state.method_name,
                "evolution_step_before_task": step_before_task,
                "evolution_step_after_task": state.step,
                "reflection_count": len(state.reflections),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            },
        }
