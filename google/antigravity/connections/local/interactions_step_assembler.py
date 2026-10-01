# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assembles GAOS Interactions step.start, step.delta, and step.stop events into SDK Steps."""

import collections
import dataclasses
from typing import Any

from google.antigravity import types
from google.antigravity.connections.local import event_processor
from google.antigravity.connections.local.local_connection_config import make_step_id
from google.antigravity.connections.local.local_connection_config import normalize_wire_path
from google.antigravity.connections.local.local_connection_config import WIRE_PATH_ARGUMENT_KEYS

_CALL_TO_RESULT_KIND: dict[str, str] = {
    "function_call": "function",
    "code_execution_call": "code_execution",
    "view_file_call": "view_file",
    "list_directory_call": "list_directory",
    "find_file_call": "find_file",
    "search_directory_call": "search_directory",
    "edit_file_call": "edit_file",
    "generate_image_call": "generate_image",
    "google_search_call": "google_search",
    "url_context_call": "url_context",
    "mcp_server_tool_call": "mcp_server_tool",
    "skill_lookup_call": "skill_lookup",
}

_RESULT_TO_CALL_KIND: dict[str, str] = {
    "function_result": "function",
    "code_execution_result": "code_execution",
    "view_file_result": "view_file",
    "list_directory_result": "list_directory",
    "find_file_result": "find_file",
    "search_directory_result": "search_directory",
    "edit_file_result": "edit_file",
    "generate_image_result": "generate_image",
    "google_search_result": "google_search",
    "url_context_result": "url_context",
    "mcp_server_tool_result": "mcp_server_tool",
    "skill_lookup_result": "skill_lookup",
}

_RESULT_KIND_DEFAULT_TOOL_NAME: dict[str, str] = {
    "code_execution": types.BuiltinTools.RUN_COMMAND.value,
    "view_file": types.BuiltinTools.VIEW_FILE.value,
    "list_directory": types.BuiltinTools.LIST_DIR.value,
    "find_file": types.BuiltinTools.FIND_FILE.value,
    "search_directory": types.BuiltinTools.SEARCH_DIR.value,
    "edit_file": types.BuiltinTools.EDIT_FILE.value,
    "generate_image": types.BuiltinTools.GENERATE_IMAGE.value,
    "google_search": types.BuiltinTools.SEARCH_WEB.value,
    "url_context": types.BuiltinTools.READ_URL_CONTENT.value,
    # skill_lookup is surfaced for SkillsConfig (skills_paths) and is not a
    # member of types.BuiltinTools.
    "skill_lookup": "skill_lookup",
}

_CALL_STEP_ENVELOPE_KEYS: frozenset[str] = frozenset({
    "type",
    "id",
    "signature",
    "timestamp",
    "description",
    "name",
    "server_name",
    "arguments",
})

_RESULT_STEP_ENVELOPE_KEYS: frozenset[str] = frozenset({
    "type",
    "call_id",
    "signature",
    "timestamp",
    "description",
    "name",
    "server_name",
    "is_error",
    "result",
})


@dataclasses.dataclass(frozen=True)
class ParsedElicitation:
  """Parsed question elicitation from an elicitation_call step."""

  elicitation_id: str
  group_key: str
  trajectory_id: str
  step_index: int
  question: types.AskQuestionEntry


@dataclasses.dataclass
class StepAssemblyResult:
  """Result of processing a step.start, step.delta, or step.stop event."""

  step: event_processor.LocalConnectionStep | None = None
  dispatch_pre: bool = False
  dispatch_post: bool = False
  step_usage: types.UsageMetadata | None = None
  trajectory_id: str = ""
  elicitation: ParsedElicitation | None = None


@dataclasses.dataclass
class _AssembledStepState:
  """Internal state for an in-flight GAOS step."""

  trajectory_id: str
  parent_trajectory_id: str
  depth: int
  step_index: int
  step_subtype: str
  step_type: types.StepType
  source: types.StepSource
  target: types.StepTarget
  description: str = ""
  content: str = ""
  thinking: str = ""
  tool_calls: list[types.ToolCall] = dataclasses.field(default_factory=list)
  is_error: bool = False
  error_message: str = ""
  is_tool_call_step: bool = False
  is_tool_result_step: bool = False


def parse_interaction_usage(
    usage_dict: dict[str, Any] | None,
) -> types.UsageMetadata | None:
  """Parses a GAOS Interaction.Usage dictionary into SDK UsageMetadata."""
  if not usage_dict or not isinstance(usage_dict, dict):
    return None
  prompt_tokens = usage_dict.get("total_input_tokens")
  cached_tokens = usage_dict.get("total_cached_tokens")
  output_tokens = usage_dict.get("total_output_tokens")
  thought_tokens = usage_dict.get("total_thought_tokens")
  total_tokens = usage_dict.get("total_tokens")
  if all(
      v is None
      for v in (
          prompt_tokens,
          cached_tokens,
          output_tokens,
          thought_tokens,
          total_tokens,
      )
  ):
    return None
  return types.UsageMetadata(
      prompt_token_count=prompt_tokens,
      cached_content_token_count=cached_tokens,
      candidates_token_count=output_tokens,
      thoughts_token_count=thought_tokens,
      total_token_count=total_tokens,
  )


def _extract_text_from_content(content_raw: Any) -> str:
  """Extracts plain text from a GAOS Content union (str, dict, or list of blocks)."""
  if isinstance(content_raw, str):
    return content_raw
  if isinstance(content_raw, dict):
    if "name" in content_raw and "type" not in content_raw:
      return f"/{content_raw['name']}"
    if content_raw.get("type") == "text":
      return str(content_raw.get("text", ""))
    if "content" in content_raw:
      return _extract_text_from_content(content_raw["content"])
    return ""
  if isinstance(content_raw, list):
    parts = []
    for item in content_raw:
      if isinstance(item, dict) and item.get("type") == "text":
        parts.append(str(item.get("text", "")))
      elif isinstance(item, str):
        parts.append(item)
    return "".join(parts)
  return ""


def _normalize_tool_args(
    args: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
  """Normalizes wire path URIs in tool arguments and returns (normalized_args, canonical_path)."""
  normalized_args = dict(args)
  canonical_path: str | None = None
  for path_key in WIRE_PATH_ARGUMENT_KEYS:
    val = normalized_args.get(path_key)
    if isinstance(val, str) and val:
      norm = normalize_wire_path(val)
      normalized_args[path_key] = norm
      if canonical_path is None:
        canonical_path = norm
  if "image_paths" in normalized_args and isinstance(
      normalized_args["image_paths"], list
  ):
    normalized_args["image_paths"] = [
        normalize_wire_path(p) if isinstance(p, str) else p
        for p in normalized_args["image_paths"]
    ]
  return normalized_args, canonical_path


def _extract_elicitation_group_key(elicitation_id: str) -> str:
  """Extracts the shared question batch prefix from `<conv>-question-<step>-<qIdx>`."""
  marker = "-question-"
  pos = elicitation_id.rfind("-")
  if marker in elicitation_id and pos > elicitation_id.find(marker):
    return elicitation_id[:pos]
  return elicitation_id


def _parse_elicitation_call(
    step_dict: dict[str, Any],
    trajectory_id: str,
    step_index: int,
) -> ParsedElicitation | None:
  """Extracts a ParsedElicitation from an elicitation_call step dict."""
  el_id = str(step_dict.get("elicitation_id", ""))
  if not el_id:
    return None
  preamble = step_dict.get("preamble") or {}
  question_text = _extract_text_from_content(preamble.get("content", []))
  mc_req = step_dict.get("multiple_choice_request") or {}
  allow_multi = bool(mc_req.get("allow_multiple_selections", False))
  options: list[types.AskQuestionOption] = []
  for idx, choice in enumerate(mc_req.get("choices") or []):
    if not isinstance(choice, dict):
      continue
    label = str(choice.get("label") or (idx + 1))
    display_text = _extract_text_from_content(choice.get("display", []))
    options.append(types.AskQuestionOption(id=label, text=display_text))

  entry = types.AskQuestionEntry(
      question=question_text,
      options=options,
      is_multi_select=allow_multi,
  )
  return ParsedElicitation(
      elicitation_id=el_id,
      group_key=_extract_elicitation_group_key(el_id),
      trajectory_id=trajectory_id,
      step_index=step_index,
      question=entry,
  )


class InteractionsStepAssembler:
  """Stateful assembler translating GAOS step.start/delta/stop events into LocalConnectionSteps."""

  def __init__(self, root_interaction_id: str = ""):
    self._root_interaction_id = root_interaction_id
    self._steps: dict[tuple[str, int], _AssembledStepState] = {}
    self._pending_calls_by_id: dict[str, types.ToolCall] = {}
    self._pending_calls_by_kind: dict[
        tuple[str, str], collections.deque[types.ToolCall]
    ] = collections.defaultdict(collections.deque)
    self._parent_trajectories: dict[str, str] = {}

  @property
  def root_interaction_id(self) -> str:
    return self._root_interaction_id

  def set_root_interaction_id(self, interaction_id: str) -> None:
    if interaction_id:
      self._root_interaction_id = interaction_id

  def record_parent_trajectory(
      self, sub_interaction_id: str, parent_interaction_id: str
  ) -> None:
    if sub_interaction_id and parent_interaction_id:
      self._parent_trajectories[sub_interaction_id] = parent_interaction_id

  def _resolve_trajectory(self, event: dict[str, Any]) -> tuple[str, str, int]:
    """Returns (trajectory_id, parent_trajectory_id, depth) for an event."""
    sub_id = str(event.get("sub_interaction_id") or "")
    if sub_id and sub_id != self._root_interaction_id:
      parent_id = self._parent_trajectories.get(
          sub_id, self._root_interaction_id
      )
      depth = 1
      curr = parent_id
      visited = {sub_id}
      while (
          curr
          and curr != self._root_interaction_id
          and curr not in visited
          and curr in self._parent_trajectories
      ):
        visited.add(curr)
        curr = self._parent_trajectories[curr]
        depth += 1
      return sub_id, parent_id, depth
    return self._root_interaction_id, "", 0

  def handle_step_start(self, event: dict[str, Any]) -> StepAssemblyResult:
    """Processes a `step.start` event dictionary."""
    traj_id, parent_traj_id, depth = self._resolve_trajectory(event)
    step_idx = int(event.get("index", 0))
    step_dict = event.get("step") or {}
    subtype = str(step_dict.get("type", ""))
    description = str(step_dict.get("description", ""))

    if subtype == "user_input":
      text = _extract_text_from_content(step_dict.get("content", ""))
      step_type = (
          types.StepType.TEXT_RESPONSE if text else types.StepType.UNKNOWN
      )
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype=subtype,
          step_type=step_type,
          source=types.StepSource.USER,
          target=types.StepTarget.USER,
          description=description,
          content=text,
      )
      self._steps[(traj_id, step_idx)] = state
      step_obj = self._build_step_object(
          state,
          status=types.StepStatus.ACTIVE,
          content_delta=text,
      )
      return StepAssemblyResult(
          step=step_obj,
          dispatch_pre=True,
          trajectory_id=traj_id,
      )

    if subtype == "thought":
      initial_thinking = _extract_text_from_content(
          step_dict.get("summary", [])
      )
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype=subtype,
          step_type=types.StepType.THINKING,
          source=types.StepSource.MODEL,
          target=types.StepTarget.USER,
          description=description,
          thinking=initial_thinking,
      )
      self._steps[(traj_id, step_idx)] = state
      step_obj = self._build_step_object(
          state,
          status=types.StepStatus.ACTIVE,
          thinking_delta=initial_thinking,
      )
      return StepAssemblyResult(
          step=step_obj,
          dispatch_pre=True,
          trajectory_id=traj_id,
      )

    if subtype == "model_output":
      initial_text = _extract_text_from_content(step_dict.get("content", []))
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype=subtype,
          step_type=types.StepType.TEXT_RESPONSE,
          source=types.StepSource.MODEL,
          target=types.StepTarget.USER,
          description=description,
          content=initial_text,
      )
      self._steps[(traj_id, step_idx)] = state
      step_obj = self._build_step_object(
          state,
          status=types.StepStatus.ACTIVE,
          content_delta=initial_text,
      )
      return StepAssemblyResult(
          step=step_obj,
          dispatch_pre=True,
          trajectory_id=traj_id,
      )

    if subtype == "elicitation_call":
      elicitation = _parse_elicitation_call(step_dict, traj_id, step_idx)
      tc = types.ToolCall(
          name=types.BuiltinTools.ASK_QUESTION.value,
          args=(
              {"question": elicitation.question.question} if elicitation else {}
          ),
          id=elicitation.elicitation_id
          if elicitation
          else make_step_id(traj_id, step_idx),
          step_id=make_step_id(traj_id, step_idx),
      )
      if tc.id:
        self._pending_calls_by_id[tc.id] = tc
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype=subtype,
          step_type=types.StepType.TOOL_CALL,
          source=types.StepSource.MODEL,
          target=types.StepTarget.USER,
          description=description,
          content=description,
          tool_calls=[tc],
          is_tool_call_step=True,
      )
      self._steps[(traj_id, step_idx)] = state
      step_obj = self._build_step_object(
          state,
          status=types.StepStatus.WAITING_FOR_USER,
      )
      return StepAssemblyResult(
          step=step_obj,
          dispatch_pre=True,
          trajectory_id=traj_id,
          elicitation=elicitation,
      )

    if subtype == "elicitation_result":
      el_id = str(step_dict.get("elicitation_id", ""))
      matched_tc = self._pending_calls_by_id.pop(el_id, None) if el_id else None
      tc = matched_tc or types.ToolCall(
          name=types.BuiltinTools.ASK_QUESTION.value,
          args={},
          id=el_id or make_step_id(traj_id, step_idx),
          step_id=make_step_id(traj_id, step_idx),
      )
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype=subtype,
          step_type=types.StepType.TOOL_CALL,
          source=types.StepSource.MODEL,
          target=types.StepTarget.USER,
          description=description,
          content=description,
          tool_calls=[tc],
          is_tool_result_step=True,
      )
      self._steps[(traj_id, step_idx)] = state
      return StepAssemblyResult(trajectory_id=traj_id)

    if subtype in _CALL_TO_RESULT_KIND:
      tc = self._build_tool_call_from_call_step(
          subtype, step_dict, traj_id, step_idx
      )
      kind = _CALL_TO_RESULT_KIND[subtype]
      if tc.id:
        self._pending_calls_by_id[tc.id] = tc
      self._pending_calls_by_kind[(traj_id, kind)].append(tc)
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype=subtype,
          step_type=types.StepType.TOOL_CALL,
          source=types.StepSource.MODEL,
          target=types.StepTarget.ENVIRONMENT,
          description=description,
          content=description,
          tool_calls=[tc],
          is_tool_call_step=True,
      )
      self._steps[(traj_id, step_idx)] = state
      step_obj = self._build_step_object(
          state,
          status=types.StepStatus.ACTIVE,
      )
      return StepAssemblyResult(
          step=step_obj,
          dispatch_pre=True,
          trajectory_id=traj_id,
      )

    if subtype in _RESULT_TO_CALL_KIND:
      tc, is_error, error_msg, result_summary = (
          self._resolve_tool_result_from_result_step(
              subtype, step_dict, traj_id, step_idx
          )
      )
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype=subtype,
          step_type=types.StepType.TOOL_CALL,
          source=types.StepSource.MODEL,
          target=types.StepTarget.ENVIRONMENT,
          description=description,
          content=result_summary or description,
          tool_calls=[tc],
          is_error=is_error,
          error_message=error_msg,
          is_tool_result_step=True,
      )
      self._steps[(traj_id, step_idx)] = state
      return StepAssemblyResult(trajectory_id=traj_id)

    # Unknown or generic step type.
    state = _AssembledStepState(
        trajectory_id=traj_id,
        parent_trajectory_id=parent_traj_id,
        depth=depth,
        step_index=step_idx,
        step_subtype=subtype,
        step_type=types.StepType.UNKNOWN,
        source=types.StepSource.MODEL,
        target=types.StepTarget.UNSPECIFIED,
        description=description,
        content=description,
    )
    self._steps[(traj_id, step_idx)] = state
    step_obj = self._build_step_object(state, status=types.StepStatus.ACTIVE)
    return StepAssemblyResult(
        step=step_obj,
        dispatch_pre=True,
        trajectory_id=traj_id,
    )

  def handle_step_delta(self, event: dict[str, Any]) -> StepAssemblyResult:
    """Processes a `step.delta` event dictionary."""
    traj_id, parent_traj_id, depth = self._resolve_trajectory(event)
    step_idx = int(event.get("index", 0))
    delta = event.get("delta") or {}
    delta_type = str(delta.get("type", ""))

    state = self._steps.get((traj_id, step_idx))
    dispatch_pre = state is None
    if state is None:
      is_thought = delta_type in ("raw_thought", "thought_summary")
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype="thought" if is_thought else "model_output",
          step_type=(
              types.StepType.THINKING
              if is_thought
              else types.StepType.TEXT_RESPONSE
          ),
          source=types.StepSource.MODEL,
          target=types.StepTarget.USER,
      )
      self._steps[(traj_id, step_idx)] = state

    if delta_type in ("raw_thought", "thought_summary") or (
        state.step_type == types.StepType.THINKING and delta_type != "text"
    ):
      content_block = delta.get("content")
      if isinstance(content_block, dict):
        delta_text = str(content_block.get("text", ""))
      else:
        delta_text = str(delta.get("text", ""))
      state.thinking += delta_text
      step_obj = self._build_step_object(
          state,
          status=types.StepStatus.ACTIVE,
          thinking_delta=delta_text,
      )
      return StepAssemblyResult(
          step=step_obj,
          dispatch_pre=dispatch_pre,
          trajectory_id=traj_id,
      )

    delta_text = str(delta.get("text", ""))
    state.content += delta_text
    if state.step_type == types.StepType.UNKNOWN and state.content:
      state.step_type = types.StepType.TEXT_RESPONSE
    step_obj = self._build_step_object(
        state,
        status=types.StepStatus.ACTIVE,
        content_delta=delta_text,
    )
    return StepAssemblyResult(
        step=step_obj,
        dispatch_pre=dispatch_pre,
        trajectory_id=traj_id,
    )

  def handle_step_stop(self, event: dict[str, Any]) -> StepAssemblyResult:
    """Processes a `step.stop` event dictionary."""
    traj_id, parent_traj_id, depth = self._resolve_trajectory(event)
    step_idx = int(event.get("index", 0))
    step_usage = parse_interaction_usage(event.get("usage"))

    state = self._steps.get((traj_id, step_idx))
    if state is None:
      state = _AssembledStepState(
          trajectory_id=traj_id,
          parent_trajectory_id=parent_traj_id,
          depth=depth,
          step_index=step_idx,
          step_subtype="model_output",
          step_type=types.StepType.UNKNOWN,
          source=types.StepSource.MODEL,
          target=types.StepTarget.USER,
      )

    # For *_call steps (including elicitation_call), `step.start` and
    # `step.stop` are emitted together when the call is initiated, while the
    # completion step is emitted later as a separate *_result step. Suppress
    # duplicate emission on `step.stop` of the call step itself.
    if state.is_tool_call_step:
      return StepAssemblyResult(
          step=None,
          step_usage=step_usage,
          trajectory_id=traj_id,
      )

    final_status = (
        types.StepStatus.ERROR if state.is_error else types.StepStatus.DONE
    )
    if (
        state.step_subtype == "model_output"
        and not state.content
        and not state.is_error
    ):
      state.step_type = types.StepType.UNKNOWN

    step_obj = self._build_step_object(state, status=final_status)
    return StepAssemblyResult(
        step=step_obj,
        dispatch_post=True,
        step_usage=step_usage,
        trajectory_id=traj_id,
    )

  def _build_step_object(
      self,
      state: _AssembledStepState,
      *,
      status: types.StepStatus,
      content_delta: str = "",
      thinking_delta: str = "",
  ) -> event_processor.LocalConnectionStep:
    """Builds a LocalConnectionStep from the current _AssembledStepState."""
    is_complete_response = (
        state.source == types.StepSource.MODEL
        and status == types.StepStatus.DONE
        and bool(state.content)
        and state.target == types.StepTarget.USER
        and state.step_type == types.StepType.TEXT_RESPONSE
    )
    return event_processor.LocalConnectionStep(
        id=make_step_id(state.trajectory_id, state.step_index),
        step_index=state.step_index,
        trajectory_id=state.trajectory_id,
        parent_trajectory_id=state.parent_trajectory_id,
        depth=state.depth,
        type=state.step_type,
        source=state.source,
        target=state.target,
        status=status,
        content=state.content,
        content_delta=content_delta,
        thinking=state.thinking,
        thinking_delta=thinking_delta,
        tool_calls=list(state.tool_calls),
        error=state.error_message,
        is_complete_response=is_complete_response,
    )

  def _build_tool_call_from_call_step(
      self,
      subtype: str,
      step_dict: dict[str, Any],
      traj_id: str,
      step_idx: int,
  ) -> types.ToolCall:
    """Constructs a normalized types.ToolCall from a GAOS *_call step dict."""
    step_id = make_step_id(traj_id, step_idx)
    call_id = str(step_dict.get("id") or step_id)
    kind = _CALL_TO_RESULT_KIND.get(subtype, subtype)
    tool_name = str(
        step_dict.get("name")
        or _RESULT_KIND_DEFAULT_TOOL_NAME.get(kind, subtype)
    )
    server_name = str(step_dict.get("server_name") or "") or None

    # Extract all non-envelope fields directly from the flattened step_dict,
    # and merge any nested `arguments` dictionary so new proto fields are
    # preserved automatically without per-field enumeration.
    raw_args: dict[str, Any] = {
        k: list(v) if isinstance(v, list) else v
        for k, v in step_dict.items()
        if k not in _CALL_STEP_ENVELOPE_KEYS
    }
    args_field = step_dict.get("arguments")
    if isinstance(args_field, dict):
      raw_args.update(args_field)

    if subtype == "code_execution_call":
      raw_args.setdefault("command_line", raw_args.pop("code", ""))
      raw_args.setdefault("language", "bash")
    elif subtype == "google_search_call":
      queries = list(raw_args.get("queries") or [])
      raw_args["queries"] = queries
      raw_args.setdefault("query", queries[0] if queries else "")
    elif subtype == "url_context_call":
      urls = list(raw_args.get("urls") or [])
      raw_args["urls"] = urls
      raw_args.setdefault("url", urls[0] if urls else "")
    elif subtype == "skill_lookup_call":
      raw_args.setdefault("operation", "")
      raw_args["requested_skill_names"] = list(
          raw_args.get("requested_skill_names") or []
      )

    norm_args, canonical_path = _normalize_tool_args(raw_args)
    return types.ToolCall(
        name=tool_name,
        args=norm_args,
        id=call_id,
        step_id=step_id,
        canonical_path=canonical_path,
        server_name=server_name,
    )

  def _resolve_tool_result_from_result_step(
      self,
      subtype: str,
      step_dict: dict[str, Any],
      traj_id: str,
      step_idx: int,
  ) -> tuple[types.ToolCall, bool, str, str]:
    """Resolves (ToolCall, is_error, error_message, summary) from a *_result step."""
    kind = _RESULT_TO_CALL_KIND[subtype]
    call_id = str(step_dict.get("call_id") or "")
    matched_tc: types.ToolCall | None = None

    if call_id and call_id in self._pending_calls_by_id:
      matched_tc = self._pending_calls_by_id.pop(call_id)
      queue = self._pending_calls_by_kind.get((traj_id, kind))
      if queue and matched_tc in queue:
        queue.remove(matched_tc)
    else:
      queue = self._pending_calls_by_kind.get((traj_id, kind))
      if queue:
        matched_tc = queue.popleft()
        if matched_tc.id:
          self._pending_calls_by_id.pop(matched_tc.id, None)

    if matched_tc is None:
      step_id = make_step_id(traj_id, step_idx)
      tool_name = str(
          step_dict.get("name")
          or _RESULT_KIND_DEFAULT_TOOL_NAME.get(kind, subtype)
      )
      server_name = str(step_dict.get("server_name") or "") or None
      matched_tc = types.ToolCall(
          name=tool_name,
          args={},
          id=call_id or step_id,
          step_id=step_id,
          server_name=server_name,
      )

    is_error = bool(step_dict.get("is_error", False))
    error_msg = ""
    summary = ""

    if subtype == "function_result":
      res_val = step_dict.get("result")
      if is_error:
        error_msg = (
            str(res_val) if res_val is not None else "Tool execution failed"
        )
      elif isinstance(res_val, str):
        summary = res_val
    elif subtype == "code_execution_result":
      output = str(step_dict.get("result", ""))
      summary = output
      if is_error:
        error_msg = output or (
            f"Command failed with exit code {step_dict.get('exit_code')}"
        )
    elif subtype == "generate_image_result":
      # Unlike other built-in tools, generate_image splits its input parameters
      # across GenerateImageCallStep (prompt, image_paths) and
      # GenerateImageResultStep (image_name, aspect_ratio) in steps.proto.
      updated_args = dict(matched_tc.args)
      for k, v in step_dict.items():
        if k not in _RESULT_STEP_ENVELOPE_KEYS:
          updated_args[k] = v
      norm_args, canonical_path = _normalize_tool_args(updated_args)
      matched_tc = matched_tc.model_copy(
          update={
              "args": norm_args,
              "canonical_path": canonical_path or matched_tc.canonical_path,
          }
      )
    elif subtype == "google_search_result":
      results = step_dict.get("result") or []
      suggestions = [
          str(r.get("search_suggestions", ""))
          for r in results
          if isinstance(r, dict) and r.get("search_suggestions")
      ]
      summary = "\n".join(suggestions)
      if is_error:
        error_msg = summary or "Web search failed"
    elif subtype == "url_context_result":
      if is_error:
        error_msg = "Reading URL content failed"
    elif subtype == "skill_lookup_result":
      err = str(step_dict.get("error_message", ""))
      if err:
        is_error = True
        error_msg = err

    return matched_tc, is_error, error_msg, summary
