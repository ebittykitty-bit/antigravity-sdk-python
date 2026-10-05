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

"""Tests for HookRunner and short-circuiting logic v2."""

from typing import Any
import unittest

from google.antigravity import types
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks


class HookRunnerTest(unittest.IsolatedAsyncioTestCase):

  async def test_dispatch_pre_turn_allow(self):

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=True)

    runner = hook_runner.HookRunner(pre_turn_hooks=[DummyPreTurnHook()])
    res, turn_context = await runner.dispatch_pre_turn("prompt")
    self.assertTrue(res.allow)
    self.assertIsInstance(turn_context, hooks.TurnContext)

  async def test_dispatch_pre_turn_deny(self):

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: types.Content
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=False, message="Denied")

    runner = hook_runner.HookRunner(pre_turn_hooks=[DummyPreTurnHook()])
    res, _ = await runner.dispatch_pre_turn("prompt")
    self.assertFalse(res.allow)
    self.assertEqual(res.message, "Denied")

  async def test_dispatch_pre_turn_multimodal_list(self):
    captured = []

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        captured.append(data)
        return hooks.HookResult(allow=True)

    runner = hook_runner.HookRunner(pre_turn_hooks=[DummyPreTurnHook()])
    res, _ = await runner.dispatch_pre_turn(["image", "text"])
    self.assertTrue(res.allow)
    self.assertEqual(captured, [["image", "text"]])
    self.assertIsInstance(runner.session_context, hooks.SessionContext)

  async def test_dispatch_pre_turn_none_normalizes_to_empty_string(self):
    captured = []

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        captured.append(data)
        return hooks.HookResult(allow=True)

    runner = hook_runner.HookRunner(pre_turn_hooks=[DummyPreTurnHook()])
    res, _ = await runner.dispatch_pre_turn(None)
    self.assertTrue(res.allow)
    self.assertEqual(captured, [""])
    self.assertIsInstance(runner.session_context, hooks.SessionContext)

  async def test_dispatch_session_start(self):
    called = False

    class DummyHook(hooks.OnSessionStartHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        nonlocal called
        called = True

    runner = hook_runner.HookRunner(on_session_start_hooks=[DummyHook()])
    await runner.dispatch_session_start()
    self.assertTrue(called)

  async def test_dispatch_session_end(self):
    called = False

    class DummyHook(hooks.OnSessionEndHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        nonlocal called
        called = True

    runner = hook_runner.HookRunner(on_session_end_hooks=[DummyHook()])
    await runner.dispatch_session_end()
    self.assertTrue(called)

  async def test_dispatch_interaction(self):

    class DummyInteractionHook(hooks.OnInteractionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        if data == "magic_question":
          return "magic_answer"
        return None

    runner = hook_runner.HookRunner(
        on_interaction_hooks=[DummyInteractionHook()]
    )
    turn_context = hooks.TurnContext(runner.session_context)

    res, answer, _ = await runner.dispatch_interaction(
        turn_context, "magic_question"
    )
    self.assertTrue(res.allow)
    self.assertEqual(answer, "magic_answer")

    res, answer, _ = await runner.dispatch_interaction(
        turn_context, "other_question"
    )
    self.assertFalse(res.allow)
    self.assertIsNone(answer)

  async def test_dispatch_pre_tool_call_decide(self):
    call_order = []

    class OrderDecideHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        call_order.append("decide")
        return hooks.HookResult(allow=True)

    runner = hook_runner.HookRunner(
        pre_tool_call_decide_hooks=[OrderDecideHook()],
    )

    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="t", args={})

    res, tool_call, _ = await runner.dispatch_pre_tool_call(
        turn_context, tool_call
    )

    self.assertTrue(res.allow)
    self.assertEqual(call_order, ["decide"])

  async def test_dispatch_pre_tool_call_modified_args(self):
    class ModifyHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        del context, data
        return hooks.HookResult(allow=True, modified_args={"cmd": "echo safe"})

    runner = hook_runner.HookRunner(
        pre_tool_call_decide_hooks=[ModifyHook()],
    )

    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="run_command", args={"cmd": "rm -rf /"})

    res, updated_tc, _ = await runner.dispatch_pre_tool_call(
        turn_context, tool_call
    )

    self.assertTrue(res.allow)
    self.assertEqual(res.modified_args, {"cmd": "echo safe"})
    self.assertEqual(updated_tc.args, {"cmd": "echo safe"})

  async def test_dispatch_pre_tool_call_chaining(self):
    observed_in_hook2 = {}

    class Hook1(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        del context, data
        return hooks.HookResult(allow=True, modified_args={"a": 10})

    class Hook2(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        del context
        observed_in_hook2.update(data.args)
        return hooks.HookResult(allow=True, modified_args={"b": 20})

    runner = hook_runner.HookRunner(
        pre_tool_call_decide_hooks=[Hook1(), Hook2()],
    )

    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="test_tool", args={"a": 1, "b": 2, "c": 3})

    res, updated_tc, _ = await runner.dispatch_pre_tool_call(
        turn_context, tool_call
    )

    self.assertTrue(res.allow)
    # Hook2 should have seen Hook1's modified "a"
    self.assertEqual(observed_in_hook2, {"a": 10, "b": 2, "c": 3})
    # Final tool_call and result should have both modifications
    self.assertEqual(res.modified_args, {"a": 10, "b": 20, "c": 3})
    self.assertEqual(updated_tc.args, {"a": 10, "b": 20, "c": 3})

  async def test_dispatch_pre_tool_call_modifying_then_allow(self):
    observed_in_hook2 = {}

    class ModifyingHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        del context, data
        return hooks.HookResult(allow=True, modified_args={"a": 10})

    class PassiveAllowHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        del context
        observed_in_hook2.update(data.args)
        return hooks.HookResult(allow=True, message="Audited by policy")

    runner = hook_runner.HookRunner(
        pre_tool_call_decide_hooks=[ModifyingHook(), PassiveAllowHook()],
    )

    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="test_tool", args={"a": 1, "b": 2, "c": 3})

    res, updated_tc, _ = await runner.dispatch_pre_tool_call(
        turn_context, tool_call
    )

    self.assertTrue(res.allow)
    self.assertEqual(observed_in_hook2, {"a": 10, "b": 2, "c": 3})
    self.assertEqual(res.message, "Audited by policy")
    self.assertEqual(res.modified_args, {"a": 10, "b": 2, "c": 3})
    self.assertEqual(updated_tc.args, {"a": 10, "b": 2, "c": 3})

  async def test_dispatch_pre_tool_call_modifying_then_deny(self):
    observed_in_hook2 = {}

    class ModifyingHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        del context, data
        return hooks.HookResult(allow=True, modified_args={"cmd": "echo safe"})

    class DenyingHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        del context
        observed_in_hook2.update(data.args)
        return hooks.HookResult(allow=False, message="Forbidden tool")

    runner = hook_runner.HookRunner(
        pre_tool_call_decide_hooks=[ModifyingHook(), DenyingHook()],
    )

    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="run_command", args={"cmd": "rm -rf /"})

    res, updated_tc, _ = await runner.dispatch_pre_tool_call(
        turn_context, tool_call
    )

    self.assertFalse(res.allow)
    self.assertEqual(observed_in_hook2, {"cmd": "echo safe"})
    self.assertEqual(res.message, "Forbidden tool")
    self.assertEqual(updated_tc.args, {"cmd": "echo safe"})

  async def test_context_scoping(self):
    runner = hook_runner.HookRunner()
    runner.session_context.set_state("session_key", "session_value")

    turn_context = hooks.TurnContext(runner.session_context)
    turn_context.set_state("turn_key", "turn_value")

    op_context = hooks.OperationContext(turn_context)
    op_context.set_state("op_key", "op_value")

    self.assertEqual(op_context.get_state("op_key"), "op_value")
    self.assertEqual(op_context.get_state("turn_key"), "turn_value")
    self.assertEqual(op_context.get_state("session_key"), "session_value")

    # Test that parent cannot access child data
    self.assertIsNone(turn_context.get_state("op_key"))
    self.assertIsNone(runner.session_context.get_state("turn_key"))

  async def test_dispatch_on_tool_error_recovery(self):

    class RecoverErrorHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return "recovered_result"

    runner = hook_runner.HookRunner(on_tool_error_hooks=[RecoverErrorHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)

    res, data = await runner.dispatch_on_tool_error(
        op_context, ValueError("Error")
    )

    self.assertTrue(res.allow)
    self.assertEqual(data, "recovered_result")

  async def test_dispatch_compaction(self):
    called_with = []

    class DummyCompactionHook(hooks.OnCompactionHook):

      async def run(self, context: hooks.HookContext, data: types.Step) -> None:
        called_with.append(data)

    runner = hook_runner.HookRunner(on_compaction_hooks=[DummyCompactionHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    step = types.Step(type=types.StepType.COMPACTION, content="compacted")

    await runner.dispatch_compaction(turn_context, step)

    self.assertEqual(len(called_with), 1)
    self.assertEqual(called_with[0].content, "compacted")

  async def test_has_hooks_includes_compaction(self):
    runner = hook_runner.HookRunner()
    self.assertFalse(runner.has_hooks)

    class DummyCompactionHook(hooks.OnCompactionHook):

      async def run(self, context: hooks.HookContext, data: types.Step) -> None:
        pass

    runner = hook_runner.HookRunner(on_compaction_hooks=[DummyCompactionHook()])
    self.assertTrue(runner.has_hooks)

  async def test_register_hook(self):
    runner = hook_runner.HookRunner()

    class DummyOnSessionStartHook(hooks.OnSessionStartHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=True)

    class DummyOnToolErrorHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return None

    class DummyOnSessionEndHook(hooks.OnSessionEndHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyOnInteractionHook(hooks.OnInteractionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        pass

    class DummyPostTurnHook(hooks.PostTurnHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyPreToolCallDecideHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=True)

    class DummyPostToolCallHook(hooks.PostToolCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyOnCompactionHook(hooks.OnCompactionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyStopHook(hooks.StopHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return types.StopHookResult()

    session_start_hook = DummyOnSessionStartHook()
    pre_turn_hook = DummyPreTurnHook()
    on_tool_error_hook = DummyOnToolErrorHook()
    session_end_hook = DummyOnSessionEndHook()
    interaction_hook = DummyOnInteractionHook()
    post_turn_hook = DummyPostTurnHook()
    decide_hook = DummyPreToolCallDecideHook()
    post_tool_call_hook = DummyPostToolCallHook()
    compaction_hook = DummyOnCompactionHook()
    stop_hook = DummyStopHook()

    runner.register_hook(session_start_hook)
    runner.register_hook(pre_turn_hook)
    runner.register_hook(on_tool_error_hook)
    runner.register_hook(session_end_hook)
    runner.register_hook(interaction_hook)
    runner.register_hook(post_turn_hook)
    runner.register_hook(decide_hook)
    runner.register_hook(post_tool_call_hook)
    runner.register_hook(compaction_hook)
    runner.register_hook(stop_hook)

    self.assertIn(session_start_hook, runner.on_session_start_hooks)
    self.assertIn(pre_turn_hook, runner.pre_turn_hooks)
    self.assertIn(on_tool_error_hook, runner.on_tool_error_hooks)
    self.assertIn(session_end_hook, runner.on_session_end_hooks)
    self.assertIn(interaction_hook, runner.on_interaction_hooks)
    self.assertIn(post_turn_hook, runner.post_turn_hooks)
    self.assertIn(decide_hook, runner.pre_tool_call_decide_hooks)

    self.assertIn(post_tool_call_hook, runner.post_tool_call_hooks)
    self.assertIn(compaction_hook, runner.on_compaction_hooks)
    self.assertIn(stop_hook, runner.stop_hooks)

    with self.assertRaises(ValueError):
      runner.register_hook("not a hook")

  async def test_register_multi_hook(self):
    runner = hook_runner.HookRunner()

    class MultiHook(hooks.OnSessionStartHook, hooks.OnSessionEndHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    my_hook = MultiHook()
    runner.register_hook(my_hook)

    self.assertIn(my_hook, runner.on_session_start_hooks)
    self.assertIn(my_hook, runner.on_session_end_hooks)

  async def test_init_with_hooks(self):
    class StartHook(hooks.OnSessionStartHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class TurnHook(hooks.PostTurnHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    start_hook = StartHook()
    turn_hook = TurnHook()
    runner = hook_runner.HookRunner(hooks=[start_hook, turn_hook])

    self.assertIn(start_hook, runner.on_session_start_hooks)
    self.assertIn(turn_hook, runner.post_turn_hooks)

  async def test_dispatch_post_turn(self):
    called = False

    class DummyHook(hooks.PostTurnHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        nonlocal called
        called = True

    runner = hook_runner.HookRunner(post_turn_hooks=[DummyHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    await runner.dispatch_post_turn(turn_context, "response")
    self.assertTrue(called)

  async def test_dispatch_post_tool_call(self):
    called = False

    class DummyHook(hooks.PostToolCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        nonlocal called
        called = True

    runner = hook_runner.HookRunner(post_tool_call_hooks=[DummyHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)
    await runner.dispatch_post_tool_call(op_context, "tool_result")
    self.assertTrue(called)

  async def test_dispatch_pre_tool_call_deny(self):

    class DummyDecideHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=False, message="Denied")

    runner = hook_runner.HookRunner(
        pre_tool_call_decide_hooks=[DummyDecideHook()]
    )
    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="t", args={})

    res, _, _ = await runner.dispatch_pre_tool_call(turn_context, tool_call)

    self.assertFalse(res.allow)
    self.assertEqual(res.message, "Denied")

  async def test_dispatch_on_tool_error_exception(self):

    class FailErrorHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        raise ValueError("Hook failed")

    runner = hook_runner.HookRunner(on_tool_error_hooks=[FailErrorHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)

    res, _ = await runner.dispatch_on_tool_error(
        op_context, ValueError("Original error")
    )

    self.assertFalse(res.allow)
    self.assertIn("Error recovery failed", res.message)

  async def test_dispatch_on_tool_error_fall_through(self):

    class NoneErrorHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return None

    runner = hook_runner.HookRunner(on_tool_error_hooks=[NoneErrorHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)

    res, data = await runner.dispatch_on_tool_error(
        op_context, ValueError("Original error")
    )

    self.assertFalse(res.allow)
    self.assertIsNone(data)

  async def test_dispatch_pre_model_call_exception(self):
    """Verifies that unknown hook types raise ValueError.

    This test ensures the register_hook method correctly rejects objects
    that do not inherit from any known hook base class.
    """
    runner = hook_runner.HookRunner()
    with self.assertRaises(ValueError):
      runner.register_hook(42)

  async def test_base_class_calls(self):
    """Verifies default pass implementations in base hook classes.

    Ensures that calling super().run() on every non-abstract hook base
    class does not raise, validating the no-op default behavior.
    """

    class DummyInspectHook(hooks.OnSessionStartHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        await super().run(context, data)

    class DummyDecideHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> types.HookResult:
        await super().run(context, data)
        return types.HookResult(allow=True)

    class DummyTransformHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        await super().run(context, data)
        return data

    class DummyInteractionHook(hooks.OnInteractionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        await super().run(context, data)
        return types.QuestionHookResult(responses=[])

    ctx = hooks.HookContext()
    await DummyInspectHook().run(ctx, None)
    await DummyDecideHook().run(ctx, None)
    await DummyDecideHook().run(ctx, ["dummy_image", "text"])
    await DummyTransformHook().run(ctx, {})
    await DummyInteractionHook().run(
        ctx, types.AskQuestionInteractionSpec(questions=[])
    )

  async def _run_stop(
      self, *stop_hooks: hooks.StopHook, args: types.StopArgs | None = None
  ) -> types.StopHookResult:
    """Helper to dispatch stop hooks with minimal boilerplate."""
    runner = hook_runner.HookRunner(stop_hooks=list(stop_hooks))
    return await runner.dispatch_stop(
        hooks.TurnContext(runner.session_context), args or types.StopArgs()
    )

  async def test_dispatch_stop_continue(self):
    """Verifies dispatch_stop returns CONTINUE when a hook decides so."""
    received_context = []
    received_data = []

    class ContinueStopHook(hooks.StopHook):

      async def run(self, context, data):
        received_context.append(context)
        received_data.append(data)
        return types.StopHookResult(
            decision=types.StopDecision.CONTINUE,
            reason="Need more work",
        )

    res = await self._run_stop(
        ContinueStopHook(),
        args=types.StopArgs(response_text="done", continuation_count=0),
    )
    self.assertEqual(res.decision, types.StopDecision.CONTINUE)
    self.assertEqual(res.reason, "Need more work")
    self.assertEqual(len(received_context), 1)
    self.assertIsInstance(received_context[0], hooks.TurnContext)
    self.assertEqual(len(received_data), 1)
    self.assertEqual(received_data[0].response_text, "done")
    self.assertEqual(received_data[0].continuation_count, 0)

  async def test_dispatch_stop_allow_stop(self):
    """Verifies dispatch_stop defaults to ALLOW_STOP when no hooks continue."""

    class AllowStopHook(hooks.StopHook):

      async def run(self, context, data):
        return types.StopHookResult(decision=types.StopDecision.ALLOW_STOP)

    res = await self._run_stop(AllowStopHook())
    self.assertEqual(res.decision, types.StopDecision.ALLOW_STOP)

  async def test_dispatch_stop_no_hooks(self):
    """Verifies dispatch_stop with no hooks returns ALLOW_STOP."""
    res = await self._run_stop()
    self.assertEqual(res.decision, types.StopDecision.ALLOW_STOP)

  async def test_dispatch_stop_short_circuits_on_first_continue(self):
    """Verifies dispatch_stop short-circuits on first CONTINUE with reason."""
    call_order = []

    class Hook1(hooks.StopHook):

      async def run(self, context, data):
        call_order.append("hook1")
        return types.StopHookResult(
            decision=types.StopDecision.CONTINUE, reason="hook1 reason"
        )

    class Hook2(hooks.StopHook):

      async def run(self, context, data):
        call_order.append("hook2")
        return types.StopHookResult(
            decision=types.StopDecision.CONTINUE, reason="hook2 reason"
        )

    res = await self._run_stop(Hook1(), Hook2())
    self.assertEqual(res.decision, types.StopDecision.CONTINUE)
    self.assertEqual(res.reason, "hook1 reason")
    self.assertEqual(call_order, ["hook1"])

  async def test_dispatch_stop_error_propagates(self):
    """Verifies dispatch_stop propagates hook exceptions."""

    class FailingStopHook(hooks.StopHook):

      async def run(self, context, data):
        raise RuntimeError("hook crashed")

    with self.assertRaisesRegex(RuntimeError, "hook crashed"):
      await self._run_stop(FailingStopHook())

  async def test_dispatch_stop_nil_result_falls_through(self):
    """Verifies dispatch_stop handles None result from hook."""

    class NilResultHook(hooks.StopHook):

      async def run(self, context, data):
        return None

    res = await self._run_stop(NilResultHook())
    self.assertEqual(res.decision, types.StopDecision.ALLOW_STOP)

  async def test_has_hooks_includes_stop(self):
    runner = hook_runner.HookRunner()
    self.assertFalse(runner.has_hooks)

    class DummyStopHook(hooks.StopHook):

      async def run(self, context, data):
        return types.StopHookResult()

    runner = hook_runner.HookRunner(stop_hooks=[DummyStopHook()])
    self.assertTrue(runner.has_hooks)


class DecoratorTest(unittest.IsolatedAsyncioTestCase):

  async def test_pre_turn_decorator(self):
    captured = []

    @hooks.pre_turn
    async def my_pre_turn(data):
      captured.append(data)
      return hooks.HookResult(allow=True)

    self.assertIsInstance(my_pre_turn, hooks.PreTurnHook)
    res = await my_pre_turn.run(hooks.HookContext(), "test_prompt")
    self.assertTrue(res.allow)
    self.assertEqual(captured, ["test_prompt"])

    captured.clear()
    res2 = await my_pre_turn("direct_call")
    self.assertTrue(res2.allow)
    self.assertEqual(captured, ["direct_call"])

    dummy_image = types.Image(data=b"fake_bytes", mime_type="image/png")
    multimodal_prompt = [dummy_image, "text_prompt"]

    captured.clear()
    res3 = await my_pre_turn.run(hooks.HookContext(), multimodal_prompt)
    self.assertTrue(res3.allow)
    self.assertEqual(captured, [multimodal_prompt])

    captured.clear()
    res4 = await my_pre_turn(multimodal_prompt)
    self.assertTrue(res4.allow)
    self.assertEqual(captured, [multimodal_prompt])

  async def test_pre_tool_call_decide_decorator(self):
    @hooks.pre_tool_call_decide
    async def my_decide(data):
      return hooks.HookResult(allow=False, message=data.name)

    self.assertIsInstance(my_decide, hooks.PreToolCallDecideHook)
    tool_call = types.ToolCall(name="my_tool", args={})
    res = await my_decide.run(hooks.HookContext(), tool_call)
    self.assertFalse(res.allow)
    self.assertEqual(res.message, "my_tool")

  async def test_on_interaction_decorator(self):
    @hooks.on_interaction
    async def my_interaction(data):
      return types.QuestionHookResult(responses=[])

    self.assertIsInstance(my_interaction, hooks.OnInteractionHook)
    res = await my_interaction.run(
        hooks.HookContext(), types.AskQuestionInteractionSpec(questions=[])
    )
    self.assertEqual(res.responses, [])

  async def test_post_turn_decorator(self):
    called_with = None

    @hooks.post_turn
    async def my_post_turn(data):
      nonlocal called_with
      called_with = data

    self.assertIsInstance(my_post_turn, hooks.PostTurnHook)
    await my_post_turn.run(hooks.HookContext(), "response_data")
    self.assertEqual(called_with, "response_data")

  async def test_on_tool_error_decorator(self):
    @hooks.on_tool_error
    async def my_tool_error(data):
      return f"recovered from {type(data).__name__}"

    self.assertIsInstance(my_tool_error, hooks.OnToolErrorHook)
    res = await my_tool_error.run(hooks.HookContext(), ValueError("error"))
    self.assertEqual(res, "recovered from ValueError")


if __name__ == "__main__":
  unittest.main()
