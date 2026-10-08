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

"""Unit tests for interactions_event_processor."""

import asyncio
from typing import Any
import unittest

from absl.testing import absltest

from google.antigravity import types
from google.antigravity.connections.local import event_processor
from google.antigravity.connections.local import interactions_event_processor
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks
from google.antigravity.tools import tool_runner


class InteractionsEventProcessorTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.sent_events: list[dict[str, Any]] = []
    self.sent_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

  async def _send_json_event(self, event: dict[str, Any]) -> None:
    self.sent_events.append(event)
    await self.sent_queue.put(event)

  async def test_step_lifecycle_and_usage_accumulation(self):
    proc = interactions_event_processor.InteractionsEventProcessor(
        send_json_event_fn=self._send_json_event,
        root_interaction_id="root-1",
    )
    proc.reset_for_turn()
    self.assertFalse(proc.is_idle.is_set())

    await proc.process_event({
        "event_type": "step.start",
        "index": 0,
        "step": {"type": "model_output"},
    })
    await proc.process_event({
        "event_type": "step.delta",
        "index": 0,
        "delta": {"type": "text", "text": "Hello "},
    })
    await proc.process_event({
        "event_type": "step.delta",
        "index": 0,
        "delta": {"type": "text", "text": "world!"},
    })
    await proc.process_event({
        "event_type": "step.stop",
        "index": 0,
        "usage": {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_thought_tokens": 2,
            "total_cached_tokens": 3,
            "total_tokens": 17,
        },
    })
    await proc.process_event({
        "event_type": "state_update",
        "state": "fully_idle",
        "reason": "max_model_calls_exceeded",
    })

    self.assertTrue(proc.is_idle.is_set())
    self.assertEqual(
        proc._last_turn_stop_reason,
        types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
    )
    self.assertEqual(proc.cumulative_usage.prompt_token_count, 10)
    self.assertEqual(proc.cumulative_usage.candidates_token_count, 5)
    self.assertEqual(proc.cumulative_usage.thoughts_token_count, 2)
    self.assertEqual(proc.cumulative_usage.cached_content_token_count, 3)
    self.assertEqual(proc.cumulative_usage.total_token_count, 17)
    self.assertEqual(proc.trajectory_usages["root-1"].total_token_count, 17)

    emitted = []
    while not proc.step_queue.empty():
      item = proc.step_queue.get_nowait()
      if item is not event_processor.IDLE_SENTINEL:
        emitted.append(item)

    self.assertEqual(len(emitted), 4)
    self.assertEqual(emitted[0].status, types.StepStatus.ACTIVE)
    self.assertEqual(emitted[-1].type, types.StepType.TEXT_RESPONSE)
    self.assertEqual(emitted[-1].content, "Hello world!")
    self.assertEqual(emitted[-1].status, types.StepStatus.DONE)

  async def test_function_invocation_executes_client_tool(self):
    def add_numbers(a: int, b: int) -> int:
      """Adds two integers."""
      return a + b

    tr = tool_runner.ToolRunner(tools=[add_numbers])
    proc = interactions_event_processor.InteractionsEventProcessor(
        send_json_event_fn=self._send_json_event,
        root_interaction_id="root-1",
        tool_runner=tr,
    )

    await proc.process_event({
        "event_type": "function_invocation",
        "call_id": "call_add_1",
        "name": "add_numbers",
        "arguments": {"a": 7, "b": 8},
    })

    resp = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp,
        {
            "event_type": "function_result",
            "call_id": "call_add_1",
            "name": "add_numbers",
            "is_error": False,
            "result": {"result": 15},
        },
    )

  async def test_hook_dispatch_all_lifecycle_hooks(self):
    events_log: list[str] = []

    @hooks.on_session_start
    async def handle_start():
      events_log.append("start")

    @hooks.pre_turn
    async def handle_pre_turn(prompt: Any):
      events_log.append(f"pre_turn:{prompt}")
      return types.HookResult(allow=True)

    @hooks.pre_tool_call_decide
    async def handle_pre_tool(tool_call: Any):
      events_log.append(f"pre_tool:{tool_call.name}:{tool_call.canonical_path}")
      return types.HookResult(
          allow=True,
          modified_args={"path": tool_call.canonical_path, "max_lines": 10},
      )

    @hooks.post_tool_call
    async def handle_post_tool(tool_result: types.ToolResult):
      events_log.append(f"post_tool:{tool_result.name}:{tool_result.result}")

    @hooks.on_tool_error
    async def handle_on_tool_error(err: types.ToolExecutionError):
      events_log.append(f"on_tool_error:{err.tool_name}:{err}")
      return "Recovered error message"

    @hooks.on_compaction
    async def handle_compaction(step: types.Step):
      events_log.append(f"compaction:{step.content}")

    @hooks.stop
    async def handle_stop(stop_args: types.StopArgs):
      events_log.append(f"stop:{stop_args.response_text}")
      return types.StopHookResult(
          decision=types.StopDecision.CONTINUE,
          reason="Continue working",
      )

    @hooks.post_turn
    async def handle_post_turn(response_text: str):
      events_log.append(f"post_turn:{response_text}")

    @hooks.on_session_end
    async def handle_end():
      events_log.append("end")

    hr = hook_runner.HookRunner(
        on_session_start_hooks=[handle_start],
        pre_turn_hooks=[handle_pre_turn],
        pre_tool_call_decide_hooks=[handle_pre_tool],
        post_tool_call_hooks=[handle_post_tool],
        on_tool_error_hooks=[handle_on_tool_error],
        on_compaction_hooks=[handle_compaction],
        stop_hooks=[handle_stop],
        post_turn_hooks=[handle_post_turn],
        on_session_end_hooks=[handle_end],
    )

    proc = interactions_event_processor.InteractionsEventProcessor(
        send_json_event_fn=self._send_json_event,
        root_interaction_id="root-1",
        hook_runner=hr,
    )

    # 1. OnSessionStart
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-start",
        "type": "on_session_start",
        "name": "OnSessionStart",
    })
    resp_start = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_start,
        {
            "event_type": "call_hook_response",
            "request_id": "req-start",
            "empty_result": {},
        },
    )

    # 2. PreTurn
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-pre-turn",
        "type": "pre_turn",
        "name": "PreTurn",
        "pre_turn_args": {
            "user_input": {
                "type": "user_input",
                "content": "original prompt",
            }
        },
    })
    resp_pre_turn = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(resp_pre_turn["event_type"], "call_hook_response")
    self.assertEqual(resp_pre_turn["request_id"], "req-pre-turn")
    self.assertEqual(
        resp_pre_turn["pre_turn_result"],
        {"decision": "allow"},
    )

    # 3. PreTool (with path normalization and modified_args preserving int)
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-pre-tool",
        "type": "pre_tool",
        "name": "PreTool",
        "pre_tool_args": {
            "tool_name": "view_file",
            "arguments_json": '{"path": "file:///tmp/test.py"}',
        },
    })
    resp_pre_tool = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(resp_pre_tool["event_type"], "call_hook_response")
    self.assertEqual(resp_pre_tool["request_id"], "req-pre-tool")
    self.assertEqual(
        resp_pre_tool["pre_tool_result"],
        {
            "decision": "allow",
            "modified_args": {"path": "/tmp/test.py", "max_lines": 10},
        },
    )
    self.assertIsInstance(
        resp_pre_tool["pre_tool_result"]["modified_args"]["max_lines"], int
    )

    # 4. PostTool
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-post-tool",
        "type": "post_tool",
        "name": "PostTool",
        "post_tool_args": {
            "tool_name": "view_file",
            "result": "file contents",
        },
    })
    resp_post_tool = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_post_tool,
        {
            "event_type": "call_hook_response",
            "request_id": "req-post-tool",
            "empty_result": {},
        },
    )

    # 5. OnToolError
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-tool-err",
        "type": "on_tool_error",
        "name": "OnToolError",
        "on_tool_error_args": {
            "tool_name": "run_command",
            "error_message": "exit 1",
        },
    })
    resp_tool_err = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_tool_err,
        {
            "event_type": "call_hook_response",
            "request_id": "req-tool-err",
            "on_tool_error_result": {
                "custom_error_message": "Recovered error message"
            },
        },
    )

    # 6. OnCompaction
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-compact",
        "type": "on_compaction",
        "name": "OnCompaction",
        "on_compaction_args": {
            "interaction_id": "root-1",
            "step_index": 5,
            "summary": "Compacted summary",
        },
    })
    resp_compact = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_compact,
        {
            "event_type": "call_hook_response",
            "request_id": "req-compact",
            "empty_result": {},
        },
    )
    compaction_step = proc.step_queue.get_nowait()
    self.assertEqual(compaction_step.type, types.StepType.COMPACTION)
    self.assertEqual(compaction_step.content, "Compacted summary")
    self.assertEqual(compaction_step.id, "root-1:5")

    # 7. Stop
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-stop",
        "type": "stop",
        "name": "Stop",
        "stop_args": {
            "response_text": "Almost done",
            "interaction_id": "root-1",
            "continuation_count": 1,
        },
    })
    resp_stop = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_stop,
        {
            "event_type": "call_hook_response",
            "request_id": "req-stop",
            "stop_result": {
                "decision": "continue",
                "reason": "Continue working",
            },
        },
    )

    # 8. PostTurn
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-post-turn",
        "type": "post_turn",
        "name": "PostTurn",
        "post_turn_args": {
            "response_text": "All done",
        },
    })
    resp_post_turn = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_post_turn,
        {
            "event_type": "call_hook_response",
            "request_id": "req-post-turn",
            "empty_result": {},
        },
    )

    # 9. OnSessionEnd + interaction.completed
    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-end",
        "type": "on_session_end",
        "name": "OnSessionEnd",
    })
    resp_end = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_end,
        {
            "event_type": "call_hook_response",
            "request_id": "req-end",
            "empty_result": {},
        },
    )
    await proc.process_event({"event_type": "interaction.completed"})
    self.assertTrue(proc.session_end_done.is_set())
    self.assertEqual(
        events_log,
        [
            "start",
            "pre_turn:original prompt",
            "pre_tool:view_file:/tmp/test.py",
            "post_tool:view_file:file contents",
            "on_tool_error:run_command:exit 1",
            "compaction:Compacted summary",
            "stop:Almost done",
            "post_turn:All done",
            "end",
        ],
    )

  async def test_elicitation_question_batch_dispatch(self):
    @hooks.on_interaction
    async def handle_interaction(
        interaction: types.AskQuestionInteractionSpec,
    ):
      self.assertEqual(len(interaction.questions), 2)
      return types.QuestionHookResult(
          responses=[
              types.QuestionResponse(selected_option_ids=["1"]),
              types.QuestionResponse(selected_option_ids=["2"]),
          ]
      )

    hr = hook_runner.HookRunner(on_interaction_hooks=[handle_interaction])

    proc = interactions_event_processor.InteractionsEventProcessor(
        send_json_event_fn=self._send_json_event,
        root_interaction_id="conv-1",
        hook_runner=hr,
    )

    # Question 0
    await proc.process_event({
        "event_type": "step.start",
        "index": 10,
        "step": {
            "type": "elicitation_call",
            "elicitation_id": "conv-1-question-3-0",
            "preamble": {"content": [{"type": "text", "text": "Language?"}]},
            "multiple_choice_request": {
                "allow_multiple_selections": False,
                "choices": [
                    {
                        "label": "1",
                        "display": [{"type": "text", "text": "Python"}],
                    },
                    {
                        "label": "2",
                        "display": [{"type": "text", "text": "Go"}],
                    },
                ],
            },
        },
    })
    await proc.process_event({
        "event_type": "step.stop",
        "index": 10,
    })
    # Question 1 in the same batch
    await proc.process_event({
        "event_type": "step.start",
        "index": 11,
        "step": {
            "type": "elicitation_call",
            "elicitation_id": "conv-1-question-3-1",
            "preamble": {"content": [{"type": "text", "text": "OS?"}]},
            "multiple_choice_request": {
                "allow_multiple_selections": False,
                "choices": [
                    {
                        "label": "1",
                        "display": [{"type": "text", "text": "Linux"}],
                    },
                    {
                        "label": "2",
                        "display": [{"type": "text", "text": "macOS"}],
                    },
                ],
            },
        },
    })
    await proc.process_event({
        "event_type": "step.stop",
        "index": 11,
    })

    ev0 = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    ev1 = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        ev0,
        {
            "event_type": "elicitation_result",
            "elicitation_id": "conv-1-question-3-0",
            "multiple_choice": {
                "selected_choice_labels": ["1"],
            },
        },
    )
    self.assertEqual(
        ev1,
        {
            "event_type": "elicitation_result",
            "elicitation_id": "conv-1-question-3-1",
            "multiple_choice": {
                "selected_choice_labels": ["2"],
            },
        },
    )

  async def test_subagent_and_error_state_updates(self):
    proc = interactions_event_processor.InteractionsEventProcessor(
        send_json_event_fn=self._send_json_event,
        root_interaction_id="conv-1",
    )
    proc.is_idle.clear()

    # Subagent state_update should record parent trajectory without setting
    # is_idle on the root turn.
    await proc.process_event({
        "event_type": "state_update",
        "sub_interaction_id": "sub-1",
        "parent_interaction_id": "conv-1",
        "state": "fully_idle",
    })
    self.assertFalse(proc.is_idle.is_set())

    # Root cancelled state_update should emit AntigravityExecutionError then
    # IDLE_SENTINEL.
    await proc.process_event({
        "event_type": "state_update",
        "state": "cancelled",
        "error": "Turn cancelled by user",
    })
    self.assertTrue(proc.is_idle.is_set())
    err_item = await proc.step_queue.get()
    self.assertIsInstance(err_item, types.AntigravityExecutionError)
    self.assertIn("Turn cancelled by user", str(err_item))
    sentinel = await proc.step_queue.get()
    self.assertIs(sentinel, event_processor.IDLE_SENTINEL)

  async def test_hook_dispatch_edge_cases_and_media_tool_result(self):
    # 1. No hook_runner returns empty_result
    proc_no_hr = interactions_event_processor.InteractionsEventProcessor(
        send_json_event_fn=self._send_json_event,
        root_interaction_id="conv-1",
        hook_runner=None,
    )
    await proc_no_hr.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-no-hr",
        "type": "on_session_start",
        "name": "OnSessionStart",
    })
    resp_no_hr = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_no_hr,
        {
            "event_type": "call_hook_response",
            "request_id": "req-no-hr",
            "empty_result": {},
        },
    )

    # 2. Deny decisions, allow_stop decision, and hook exceptions
    @hooks.pre_turn
    async def deny_turn(prompt: Any):
      del prompt
      return types.HookResult(allow=False, message="turn blocked")

    @hooks.pre_tool_call_decide
    async def deny_tool(tool_call: Any):
      del tool_call
      return types.HookResult(allow=False, message="tool blocked")

    @hooks.stop
    async def allow_stop_hook(stop_args: types.StopArgs):
      del stop_args
      return types.StopHookResult(decision=types.StopDecision.ALLOW_STOP)

    @hooks.post_turn
    async def raising_post_turn(response_text: str):
      del response_text
      raise RuntimeError("boom")

    def make_chart() -> types.Image:
      """Returns an Image object."""
      return types.Image(data=b"pngbytes", mime_type="image/png")

    hr = hook_runner.HookRunner(
        pre_turn_hooks=[deny_turn],
        pre_tool_call_decide_hooks=[deny_tool],
        stop_hooks=[allow_stop_hook],
        post_turn_hooks=[raising_post_turn],
    )
    tr = tool_runner.ToolRunner(tools=[make_chart])
    proc = interactions_event_processor.InteractionsEventProcessor(
        send_json_event_fn=self._send_json_event,
        root_interaction_id="conv-1",
        hook_runner=hr,
        tool_runner=tr,
    )

    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-deny-turn",
        "type": "pre_turn",
        "name": "PreTurn",
        "pre_turn_args": {"user_input": {"type": "user_input", "content": "x"}},
    })
    resp_deny_turn = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_deny_turn["pre_turn_result"],
        {"decision": "deny", "reason": "turn blocked"},
    )

    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-deny-tool",
        "type": "pre_tool",
        "name": "PreTool",
        "pre_tool_args": {
            "tool_name": "run_command",
            "arguments_json": '{"CommandLine": "rm -rf /"}',
        },
    })
    resp_deny_tool = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_deny_tool["pre_tool_result"],
        {"decision": "deny", "reason": "tool blocked"},
    )

    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-allow-stop",
        "type": "stop",
        "name": "Stop",
        "stop_args": {"response_text": "Done"},
    })
    resp_allow_stop = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        resp_allow_stop["stop_result"],
        {"decision": "allow_stop"},
    )

    await proc.process_event({
        "event_type": "call_hook_request",
        "request_id": "req-err",
        "type": "post_turn",
        "name": "PostTurn",
        "post_turn_args": {"response_text": "Done"},
    })
    resp_err = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertIn("boom", resp_err["error_message"])

    # 3. Media tool result sanitization
    await proc.process_event({
        "event_type": "function_invocation",
        "call_id": "call-chart-1",
        "name": "make_chart",
        "arguments": {},
    })
    resp_chart = await asyncio.wait_for(self.sent_queue.get(), timeout=2.0)
    self.assertEqual(resp_chart["event_type"], "function_result")
    self.assertEqual(resp_chart["call_id"], "call-chart-1")
    self.assertFalse(resp_chart["is_error"])
    self.assertEqual(resp_chart["result"], {"result": None})


if __name__ == "__main__":
  absltest.main()
