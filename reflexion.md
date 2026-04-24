# Reflexion v1 Status

## Implemented

- Added a Reflexion-only online evolution path for CAB.
- Added `ReflexionState` in `src/cab_evaluation/evolution/evo_state.py` with:
  - `step`
  - ordered `reflections`
  - `checkpoint_steps`
  - bounded prompt rendering
  - JSON save/load
- Added a normalized `ReflectionUpdatePayload` built from `GenerationResult`:
  - issue id/title/body
  - final maintainer answer
  - conversation history
  - satisfaction status/reason
  - satisfaction conditions
  - exploration log
- Added maintainer-only Reflexion memory injection:
  - current reflections are inserted into the maintainer system prompt
  - user and judge are unchanged
  - OpenHands/Kiro maintainer prompt builders were also made compatible
- Added `ReflexionWorkflow` in `src/cab_evaluation/workflows/reflexion_workflow.py`:
  - runs CAB tasks sequentially
  - injects current reflection memory before each task
  - generates one new reflection after each completed task
  - saves checkpoint states at configured steps
  - writes generation results plus evolution metadata
- Added CLI entrypoint:
  - `python -m cab_evaluation.cli reflexion-dataset ...`
- Added initial tests in `test/test_reflexion.py`.

## Aligned with Official Reflexion Repo

- Reflection memory is treated as a simple ordered list of natural-language reflections.
- Prompt injection now uses a Reflexion-style header plus:
  - `Reflections:`
  - bullet list format
- Reflection generation prompt now follows the official pattern more closely:
  - diagnose possible reason for failure
  - produce a concise, high-level plan
  - return reflection text only

## Not Yet Aligned with Official Reflexion Repo

- No few-shot reflection examples are used yet.
- Reflection update prompt is still hardcoded in Python instead of loaded from a dedicated prompt/template file.
- We only implement the `REFLEXION` mode; official alternatives like `LAST_ATTEMPT` and `LAST_ATTEMPT_AND_REFLEXION` are not implemented.
- The previous trial is passed as a normalized CAB JSON payload, not as an official-style scratchpad / trajectory text block.
- No explicit memory window matching the official repo behavior (for example, fixed last-`k` reflections) beyond the current char-bounded rendering.
- No official-format reflection logs or trial transcripts are saved yet; only CAB generation output plus serialized Reflexion state.

## Validation Status

- `python -m compileall src/cab_evaluation` passes.
- Dynamic tests were written, but full local execution still depends on this environment having the package test/runtime deps installed (notably `pytest` and `docker`).
