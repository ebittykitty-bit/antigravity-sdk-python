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

"""Layer 1 API for Antigravity SDK."""

from collections.abc import Callable, Sequence
import contextlib
import inspect
import logging
import os
import pathlib
import re
import tempfile
from typing import Any, ClassVar, cast

from google.antigravity import beta as beta_lib
from google.antigravity import types
from google.antigravity import workflows as workflow_lib
from google.antigravity.connections import connection as connection_module
from google.antigravity.conversation import conversation as conversation_lib
from google.antigravity.hooks import hook_runner
from google.antigravity.tools import tool_context
from google.antigravity.tools import tool_runner
from google.antigravity.triggers import trigger_runner


__all__ = ["Agent"]


def _sanitize_workflow_name(
    fn_or_def: Callable[[], Any] | workflow_lib.WorkflowDefinition,
) -> tuple[str, str]:
  """Extracts the raw and filesystem-safe name for a workflow function."""
  raw_name = str(
      getattr(
          fn_or_def,
          "name",
          getattr(fn_or_def, "__name__", "workflow"),
      )
  )
  safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name).strip("_") or "workflow"
  return raw_name, safe_name


def _resolve_workflow_script_path(
    script_path: str | os.PathLike[str],
    workspaces: Sequence[str | os.PathLike[str]] | None,
    description: str,
) -> tuple[str, str]:
  """Resolves and validates an existing `.py` workflow script on disk."""
  raw_path = os.fspath(script_path)
  if not raw_path or not raw_path.strip():
    raise ValueError("run_workflow() requires a non-empty script_path.")
  path_obj = pathlib.Path(raw_path).expanduser()
  if not path_obj.is_absolute() and workspaces:
    candidate = (
        pathlib.Path(os.fspath(workspaces[0])).expanduser().resolve() / path_obj
    )
    if candidate.is_file():
      path_obj = candidate
  resolved_path = str(path_obj.resolve())
  if not pathlib.Path(resolved_path).is_file():
    raise FileNotFoundError(f"Workflow script not found: {resolved_path}")
  script_source = pathlib.Path(resolved_path).read_text(encoding="utf-8")
  workflow_lib.validate_workflow_source(script_source)
  effective_description = (
      description or f"Run workflow {pathlib.Path(resolved_path).name}"
  )
  return resolved_path, effective_description


def _materialize_workflow_function(
    fn_or_def: Callable[[], Any] | workflow_lib.WorkflowDefinition,
    workspaces: Sequence[str | os.PathLike[str]] | None,
    description: str,
) -> tuple[str, str, str]:
  """Extracts a workflow function's source and writes it to a temporary `.py` file."""
  script_source = workflow_lib.extract_workflow_source(fn_or_def)
  wf_name, safe_wf_name = _sanitize_workflow_name(fn_or_def)
  wf_doc = (
      fn_or_def.description
      if isinstance(fn_or_def, workflow_lib.WorkflowDefinition)
      else (inspect.getdoc(fn_or_def) or "").strip()
  )
  effective_description = description or wf_doc or f"Run workflow {wf_name}"
  temp_dir: str | None = None
  if workspaces:
    ws_candidate = pathlib.Path(os.fspath(workspaces[0])).expanduser().resolve()
    if ws_candidate.is_dir() and os.access(ws_candidate, os.W_OK):
      temp_dir = str(ws_candidate)
  try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        prefix=f".agy_workflow_{safe_wf_name}_",
        dir=temp_dir,
        delete=False,
    ) as tmp_file:
      tmp_file.write(script_source)
      temp_script_path = tmp_file.name
  except PermissionError:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        prefix=f".agy_workflow_{safe_wf_name}_",
        dir=None,
        delete=False,
    ) as tmp_file:
      tmp_file.write(script_source)
      temp_script_path = tmp_file.name
  resolved_path = str(pathlib.Path(temp_script_path).resolve())
  return resolved_path, effective_description, temp_script_path


class Agent:
  """High-level Agent API for simplified interaction."""

  beta: ClassVar[beta_lib.BetaNamespace] = beta_lib.BetaNamespace()

  def __init__(self, config: connection_module.AgentConfig):
    """Initializes the Agent.

    Args:
        config: Declarative agent configuration.
    """
    self._config = config.model_copy(deep=True)
    if self._config.response_schema:
      # The response_schema is validated/stringified in AgentConfig.
      self._config.capabilities.finish_tool_schema_json = cast(
          str, self._config.response_schema
      )
    self._strategy = None
    self._conversation = None
    self._tool_runner = None
    self._hook_runner = None
    self._trigger_runner = None
    # Use the original config (not self._config) for hooks and triggers:
    # model_copy(deep=True) creates new objects, breaking reference equality
    # for user-provided hooks/triggers. The list() snapshot prevents the
    # caller from mutating our copy, while preserving object identity.
    self._pending_hooks = list(config.hooks)
    self._pending_triggers = list(config.triggers)
    self._exit_stack = contextlib.AsyncExitStack()

  async def __aenter__(self) -> "Agent":
    """Starts the agent session.

    Returns:
        The started Agent instance.
    """
    logging.info("Starting Agent session")
    try:
      self._hook_runner = hook_runner.HookRunner(hooks=self._pending_hooks)
      self._pending_hooks.clear()

      # Apply policies
      active_policies = list(self._config.policies)
      cfg = self._config.capabilities
      read_only_tools = set(types.BuiltinTools.read_only()) | set(
          types.BuiltinTools.deprecated()
      )
      active_tools = connection_module.resolve_active_tools(cfg)
      has_write_tools = bool(active_tools - read_only_tools)
      has_mcp_servers = bool(self._config.mcp_servers)
      has_tool_decide_hook = bool(self._hook_runner.pre_tool_call_decide_hooks)

      if (
          (has_write_tools or has_mcp_servers)
          and not active_policies
          and not has_tool_decide_hook
      ):
        raise ValueError(
            "Write tools or MCP servers are enabled without a safety policy. "
            "Add policies=[policy.allow_all()] to approve all tool calls, "
            "or policies=[policy.deny_all(), policy.allow('tool_name')] "
            "to selectively allow specific tools."
        )

      self._tool_runner = tool_runner.ToolRunner(
          tools=self._config._get_all_custom_tools()
      )

      self._strategy = self._config.create_strategy(
          tool_runner=self._tool_runner,
          hook_runner=self._hook_runner,
      )

      logging.info("Starting connection and creating conversation...")
      self._conversation = await self._exit_stack.enter_async_context(
          conversation_lib.Conversation.create(self._strategy)
      )

      # Start triggers via TriggerRunner.
      if self._pending_triggers:
        logging.info("Starting triggers...")
        self._trigger_runner = await self._exit_stack.enter_async_context(
            trigger_runner.TriggerRunner(
                triggers=list(self._pending_triggers),
                connection=self.conversation.connection,
            )
        )
        self._pending_triggers.clear()

      # Wire ToolContext into ToolRunner so tools can access
      # conversation capabilities (same pattern as TriggerRunner).
      if self._tool_runner:
        ctx = tool_context.ToolContext(self.conversation)
        self._tool_runner.set_context(ctx)

      return self
    except Exception:
      logging.exception("Failed to start Agent session, cleaning up...")
      await self._exit_stack.aclose()
      raise

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Stops the agent session.

    Args:
        exc_type: The exception type, if any.
        exc_val: The exception value, if any.
        exc_tb: The traceback, if any.

    Returns:
        True if the exception was suppressed, False or None otherwise.
    """
    logging.info("Stopping Agent session")
    return await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

  async def chat(self, prompt: types.Content) -> types.ChatResponse:
    """Sends a prompt and returns the final response.

    Args:
        prompt: The user prompt or content to send.

    Returns:
        The final response from the agent.

    Raises:
        ValueError: If prompt is None, an empty or whitespace-only string, or an
          empty sequence / sequence containing only empty or whitespace strings.
    """
    if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
      raise ValueError(
          f"chat() requires a non-empty message string. Got: {prompt!r}"
      )
    if (
        isinstance(prompt, Sequence)
        and not isinstance(prompt, str)
        and (
            not prompt
            or all(isinstance(p, str) and not p.strip() for p in prompt)
        )
    ):
      raise ValueError(
          f"chat() requires non-empty message content. Got: {prompt!r}"
      )
    return await self.conversation.chat(prompt)

  @beta_lib.beta
  async def run_workflow(
      self,
      workflow: (
          Callable[[], Any] | workflow_lib.WorkflowDefinition | None
      ) = None,
      script_path: str | os.PathLike[str] | None = None,
      description: str = "",
  ) -> types.WorkflowResult:
    """Executes a `@workflows.define` function or a `.py` workflow script.

    Exactly one of `workflow` or `script_path` must be provided.

    Args:
      workflow: A `@workflows.define` workflow definition or a zero-argument
        Python function (`async def` or `def`) using `workflows` primitives
        (`phase`, `log`, `agent`, `parallel`, `pipeline`). Mutually exclusive
        with `script_path`.
      script_path: Path to an existing `.py` workflow script on disk. Mutually
        exclusive with `workflow`.
      description: Optional human-readable description of the workflow run.

    Returns:
      A `WorkflowResult` (subclass of `ChatResponse`) containing the executed
      script path, description, formatted workflow output (`output`), final
      assistant `response_text`, and turn-level `ChatResponse` accessors.

    Raises:
      RuntimeError: If the agent session has not been started.
      ValueError: If `BuiltinTools.RUN_WORKFLOW` is disabled on this Agent, if
        both or neither of `workflow` and `script_path` are provided, or if
        `script_path` is an empty string.
      FileNotFoundError: If `script_path` is provided and does not exist.
      workflow_lib.WorkflowError: If the workflow script or function fails AST
        validation.
      types.ToolExecutionError: If the `run_workflow` step terminates with an
        error.
    """
    if (workflow is None) == (script_path is None):
      raise ValueError(
          "run_workflow() requires exactly one of `workflow` or `script_path`."
      )

    conv = self.conversation
    active_tools = connection_module.resolve_active_tools(
        self._config.capabilities
    )
    if (
        not self._config.capabilities.enable_subagents
        or types.BuiltinTools.RUN_WORKFLOW not in active_tools
    ):
      raise ValueError(
          "BuiltinTools.RUN_WORKFLOW is not enabled on this Agent. Ensure"
          " enable_subagents=True and BuiltinTools.RUN_WORKFLOW is not"
          " disabled."
      )

    temp_script_path: str | None = None
    try:
      if script_path is not None:
        resolved_path, effective_description = _resolve_workflow_script_path(
            script_path, self._config.workspaces, description
        )
      else:
        assert workflow is not None
        resolved_path, effective_description, temp_script_path = (
            _materialize_workflow_function(
                workflow, self._config.workspaces, description
            )
        )

      history_start = len(conv.history)
      prompt = (
          f"Run the workflow script at `{resolved_path}`"
          f" ({effective_description}) using the `run_workflow` tool."
      )
      chat_resp = await self.chat(prompt)
      response_text = await chat_resp.text()

      workflow_step: types.Step | None = None
      for step in reversed(conv.history[history_start:]):
        if step.depth != 0 or step.parent_trajectory_id:
          continue
        if step.workflow_progress is not None or any(
            tc.name == types.BuiltinTools.RUN_WORKFLOW.value
            for tc in step.tool_calls
        ):
          workflow_step = step
          break

      if workflow_step is None:
        raise types.ToolExecutionError(
            tool_name=types.BuiltinTools.RUN_WORKFLOW.value,
            message="The model did not invoke the run_workflow tool.",
        )

      if workflow_step.status == types.StepStatus.ERROR:
        raise types.ToolExecutionError(
            tool_name=types.BuiltinTools.RUN_WORKFLOW.value,
            message=workflow_step.error or "Workflow execution failed.",
        )

      prog = (
          workflow_step.workflow_progress
          if workflow_step.workflow_progress is not None
          else types.WorkflowProgress()
      )
      return types.WorkflowResult._from_chat_response(  # pylint: disable=protected-access
          chat_resp,
          script_path=prog.script_path or resolved_path,
          script=prog.script,
          description=prog.description or effective_description,
          output=prog.output,
          response_text=response_text,
      )
    finally:
      if temp_script_path is not None:
        with contextlib.suppress(OSError):
          os.unlink(temp_script_path)

  @property
  def is_started(self) -> bool:
    """Whether the agent session has been started."""
    return self._conversation is not None

  @property
  def conversation(self) -> conversation_lib.Conversation:
    """Returns the active Conversation session.

    Use this for advanced session introspection: history, turn count,
    compaction indices, usage, or direct send/receive_steps control.
    For most use cases, prefer chat() instead.

    Raises:
      RuntimeError: If the agent session has not been started.
    """
    if not self._conversation:
      raise RuntimeError(
          "Agent session not started. Use 'async with Agent(...)'."
      )
    return self._conversation

  @property
  def conversation_id(self) -> str | None:
    """Returns the conversation identifier assigned by the runtime.

    Available after the session has started and at least one message has
    been exchanged.  Pass this value back via AgentConfig.conversation_id
    to resume from a saved session.  Returns None before the session starts.
    """
    if not self._conversation:
      return None
    return self._conversation.conversation_id or None

  @property
  def sandbox_status(self) -> types.SandboxStatus | None:
    """Returns the OS command sandbox status reported by the harness.

    When ``enable_sandbox`` was requested but ``sandbox_status.available`` is
    False, run_command executed unsandboxed. Application authors can inspect
    this to surface fallback UX. Returns None before the session starts or when
    the harness did not report a status.
    """
    if not self._conversation:
      return None
    return self._conversation.sandbox_status
