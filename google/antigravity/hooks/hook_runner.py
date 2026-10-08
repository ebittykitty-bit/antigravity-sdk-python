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

"""Manages registration and execution of Antigravity SDK hooks."""

from collections.abc import Sequence
import logging
from typing import Any

from google.antigravity import types
from google.antigravity.hooks import hooks as hooks_base


# Maps each hook type to the private attribute name on HookRunner that
# holds its registration list. Order matches the isinstance precedence
# of the original elif chain.
_HOOK_TYPE_REGISTRY: list[tuple[type, str]] = [
    (hooks_base.OnSessionStartHook, '_on_session_start_hooks'),
    (hooks_base.OnSessionEndHook, '_on_session_end_hooks'),
    (hooks_base.PreTurnHook, '_pre_turn_hooks'),
    (hooks_base.PostTurnHook, '_post_turn_hooks'),
    (hooks_base.PreToolCallDecideHook, '_pre_tool_call_decide_hooks'),
    (hooks_base.PostToolCallHook, '_post_tool_call_hooks'),
    (hooks_base.OnToolErrorHook, '_on_tool_error_hooks'),
    (hooks_base.OnInteractionHook, '_on_interaction_hooks'),
    (hooks_base.OnCompactionHook, '_on_compaction_hooks'),
    (hooks_base.StopHook, '_stop_hooks'),
    (hooks_base._PreStepHook, '_pre_step_hooks_list'),  # pylint: disable=protected-access
    (hooks_base._PostStepHook, '_post_step_hooks_list'),  # pylint: disable=protected-access
]


class HookRunner:
  """Manages collections of specific hook types and dispatches events."""

  def __init__(
      self,
      hooks: Sequence[Any] | None = None,
      *,
      on_session_start_hooks: list[hooks_base.OnSessionStartHook] | None = None,
      on_session_end_hooks: list[hooks_base.OnSessionEndHook] | None = None,
      pre_turn_hooks: list[hooks_base.PreTurnHook] | None = None,
      post_turn_hooks: list[hooks_base.PostTurnHook] | None = None,
      pre_tool_call_decide_hooks: (
          list[hooks_base.PreToolCallDecideHook] | None
      ) = None,
      post_tool_call_hooks: list[hooks_base.PostToolCallHook] | None = None,
      on_tool_error_hooks: list[hooks_base.OnToolErrorHook] | None = None,
      on_interaction_hooks: list[hooks_base.OnInteractionHook] | None = None,
      on_compaction_hooks: list[hooks_base.OnCompactionHook] | None = None,
      stop_hooks: list[hooks_base.StopHook] | None = None,
      _pre_step_hooks: list[hooks_base._PreStepHook] | None = None,  # pylint: disable=invalid-name,protected-access
      _post_step_hooks: list[hooks_base._PostStepHook] | None = None,  # pylint: disable=invalid-name,protected-access
  ):
    self._on_session_start_hooks = on_session_start_hooks or []
    self._on_session_end_hooks = on_session_end_hooks or []
    self._pre_turn_hooks = pre_turn_hooks or []
    self._post_turn_hooks = post_turn_hooks or []
    self._pre_tool_call_decide_hooks = pre_tool_call_decide_hooks or []
    self._post_tool_call_hooks = post_tool_call_hooks or []
    self._on_tool_error_hooks = on_tool_error_hooks or []
    self._on_interaction_hooks = on_interaction_hooks or []
    self._on_compaction_hooks = on_compaction_hooks or []
    self._stop_hooks = stop_hooks or []
    self._pre_step_hooks_list = _pre_step_hooks or []
    self._post_step_hooks_list = _post_step_hooks or []

    self.session_context = hooks_base.SessionContext()

    if hooks:
      for hook in hooks:
        self.register_hook(hook)

  @property
  def has_hooks(self) -> bool:
    """Returns True if any hooks are registered."""
    return any((
        self._on_session_start_hooks,
        self._on_session_end_hooks,
        self._pre_turn_hooks,
        self._post_turn_hooks,
        self._pre_tool_call_decide_hooks,
        self._post_tool_call_hooks,
        self._on_tool_error_hooks,
        self._on_interaction_hooks,
        self._on_compaction_hooks,
        self._stop_hooks,
        self._pre_step_hooks_list,
        self._post_step_hooks_list,
    ))

  @property
  def on_session_start_hooks(self) -> tuple[hooks_base.OnSessionStartHook, ...]:
    return tuple(self._on_session_start_hooks)

  @property
  def on_session_end_hooks(self) -> tuple[hooks_base.OnSessionEndHook, ...]:
    return tuple(self._on_session_end_hooks)

  @property
  def pre_turn_hooks(self) -> tuple[hooks_base.PreTurnHook, ...]:
    return tuple(self._pre_turn_hooks)

  @property
  def post_turn_hooks(self) -> tuple[hooks_base.PostTurnHook, ...]:
    return tuple(self._post_turn_hooks)

  @property
  def pre_tool_call_decide_hooks(
      self,
  ) -> tuple[hooks_base.PreToolCallDecideHook, ...]:
    return tuple(self._pre_tool_call_decide_hooks)

  @property
  def post_tool_call_hooks(self) -> tuple[hooks_base.PostToolCallHook, ...]:
    return tuple(self._post_tool_call_hooks)

  @property
  def on_tool_error_hooks(self) -> tuple[hooks_base.OnToolErrorHook, ...]:
    return tuple(self._on_tool_error_hooks)

  @property
  def on_interaction_hooks(self) -> tuple[hooks_base.OnInteractionHook, ...]:
    return tuple(self._on_interaction_hooks)

  @property
  def on_compaction_hooks(self) -> tuple[hooks_base.OnCompactionHook, ...]:
    return tuple(self._on_compaction_hooks)

  @property
  def stop_hooks(self) -> tuple[hooks_base.StopHook, ...]:
    return tuple(self._stop_hooks)

  @property
  def _pre_step_hooks(self) -> tuple[hooks_base._PreStepHook, ...]:
    return tuple(self._pre_step_hooks_list)

  @property
  def _post_step_hooks(self) -> tuple[hooks_base._PostStepHook, ...]:
    return tuple(self._post_step_hooks_list)

  def register_hook(self, hook: Any) -> None:
    """Registers a hook by inferring its type.

    Args:
      hook: The hook to register.

    Raises:
      ValueError: If the hook type is unknown.
    """
    registered = False
    for hook_type, attr_name in _HOOK_TYPE_REGISTRY:
      if isinstance(hook, hook_type):
        getattr(self, attr_name).append(hook)
        registered = True
    if not registered:
      raise ValueError(f'Unknown hook type: {type(hook)}')

  # Session
  async def dispatch_session_start(self) -> None:
    """Dispatches session start events to all registered hooks."""
    for hook in self._on_session_start_hooks:
      await hook.run(context=self.session_context, data=None)

  async def dispatch_session_end(self) -> None:
    """Dispatches session end events to all registered hooks."""
    for hook in self._on_session_end_hooks:
      await hook.run(context=self.session_context, data=None)

  # Turn
  async def dispatch_pre_turn(
      self, prompt: types.Content | None
  ) -> tuple[hooks_base.HookResult, hooks_base.TurnContext]:
    """Dispatches pre-turn events.

    Args:
      prompt: The user prompt, which may be text, multimodal content, or None.

    Returns:
      A tuple of (HookResult, TurnContext).
    """
    prompt = prompt or ''
    turn_context = hooks_base.TurnContext(self.session_context)
    for hook in self._pre_turn_hooks:
      res = await hook.run(context=turn_context, data=prompt)
      if not res.allow:
        return res, turn_context
    return hooks_base.HookResult(allow=True), turn_context

  async def dispatch_post_turn(
      self, turn_context: hooks_base.TurnContext, response: str
  ) -> None:
    """Dispatches post-turn events.

    Args:
      turn_context: The current turn context.
      response: The model response text.
    """
    for hook in self._post_turn_hooks:
      await hook.run(context=turn_context, data=response)

  # Tool
  async def dispatch_pre_tool_call(
      self,
      turn_context: hooks_base.TurnContext,
      tool_call: types.ToolCall,
  ) -> tuple[
      hooks_base.HookResult, types.ToolCall, hooks_base.OperationContext
  ]:
    """Dispatches pre-tool call events.

    Args:
      turn_context: The current turn context.
      tool_call: The tool call to evaluate.

    Returns:
      A tuple of (HookResult, ToolCall, OperationContext).
    """
    op_context = hooks_base.OperationContext(turn_context)
    current_tool_call = tool_call
    last_allow_result = hooks_base.HookResult(allow=True)
    had_modifications = False

    for hook in self._pre_tool_call_decide_hooks:
      res = await hook.run(context=op_context, data=current_tool_call)
      if not res.allow:
        return res, current_tool_call, op_context

      last_allow_result = res
      if res.modified_args is not None:
        merged_args = dict(current_tool_call.args)
        merged_args.update(res.modified_args)
        current_tool_call = current_tool_call.model_copy(
            update={'args': merged_args}
        )
        had_modifications = True

    if had_modifications:
      last_allow_result = last_allow_result.model_copy(
          update={'modified_args': dict(current_tool_call.args)}
      )

    return last_allow_result, current_tool_call, op_context

  async def dispatch_post_tool_call(
      self, op_context: hooks_base.OperationContext, result: Any
  ) -> None:
    """Dispatches post-tool call events.

    Args:
      op_context: The current operation context.
      result: The result of the tool call.
    """
    for hook in self._post_tool_call_hooks:
      await hook.run(context=op_context, data=result)

  async def dispatch_on_tool_error(
      self, op_context: hooks_base.OperationContext, error: Exception
  ) -> tuple[hooks_base.HookResult, Any]:
    """Dispatches tool error events.

    Args:
      op_context: The current operation context.
      error: The raised exception.

    Returns:
      A tuple of (HookResult, adjusted_error).
    """
    for hook in self._on_tool_error_hooks:
      try:
        res = await hook.run(context=op_context, data=error)
        if res is not None:
          return hooks_base.HookResult(allow=True), res
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Critical failure in OnToolErrorHook")
        return (
            hooks_base.HookResult(
                allow=False, message=f"Error recovery failed: {e}"
            ),
            None,
        )
    return hooks_base.HookResult(allow=False), None

  # Interaction
  async def dispatch_interaction(
      self, turn_context: hooks_base.TurnContext, interaction_spec: Any
  ) -> tuple[hooks_base.HookResult, Any, hooks_base.OperationContext]:
    """Dispatches interaction events.

    Args:
      turn_context: The current turn context.
      interaction_spec: The spec for the requested interaction.

    Returns:
      A tuple of (HookResult, response, OperationContext).
    """
    op_context = hooks_base.OperationContext(turn_context)
    for hook in self._on_interaction_hooks:
      res = await hook.run(context=op_context, data=interaction_spec)
      if res is not None:
        return hooks_base.HookResult(allow=True), res, op_context
    return (
        hooks_base.HookResult(
            allow=False, message="No interaction hook handled the request"
        ),
        None,
        op_context,
    )

  # Compaction
  async def dispatch_compaction(
      self, turn_context: hooks_base.TurnContext, data: types.Step
  ) -> None:
    """Dispatches compaction events.

    Args:
      turn_context: The current turn context.
      data: Data about the compaction event.
    """
    op_context = hooks_base.OperationContext(turn_context)
    for hook in self._on_compaction_hooks:
      await hook.run(context=op_context, data=data)

  async def dispatch_stop(
      self,
      turn_context: hooks_base.TurnContext,
      args: types.StopArgs,
  ) -> types.StopHookResult:
    """Dispatches stop hook events.

    Iterates stop hooks sequentially and short-circuits on the first
    `StopDecision.CONTINUE` decision with a non-empty reason.

    Args:
      turn_context: The current turn context.
      args: The stop arguments containing response text, trajectory ID,
        continuation count, stop reason, and error message.

    Returns:
      A StopHookResult with the decision and optional reason.
    """
    for hook in self._stop_hooks:
      res = await hook.run(context=turn_context, data=args)
      if res is not None and res.decision == types.StopDecision.CONTINUE:
        return res
    return types.StopHookResult(decision=types.StopDecision.ALLOW_STOP)

  # Telemetry Internal Step Hooks
  async def dispatch_pre_step(
      self, turn_context: hooks_base.TurnContext, step: types.Step
  ) -> None:
    """Dispatches internal pre-step observability events."""
    for hook in self._pre_step_hooks:
      await hook.run(context=turn_context, data=step)

  async def dispatch_post_step(
      self, turn_context: hooks_base.TurnContext, step: types.Step
  ) -> None:
    """Dispatches internal post-step observability events."""
    for hook in self._post_step_hooks:
      await hook.run(context=turn_context, data=step)
