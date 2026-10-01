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

"""Unit tests for interactions_step_assembler."""

from absl.testing import absltest
from google.antigravity import types
from google.antigravity.connections.local import interactions_step_assembler


class InteractionsStepAssemblerTest(absltest.TestCase):

  def test_user_input_step_lifecycle(self):
    assembler = interactions_step_assembler.InteractionsStepAssembler(
        root_interaction_id="root-1"
    )
    res_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 0,
        "step": {"type": "user_input", "content": "Hello agent"},
    })
    self.assertIsNotNone(res_start.step)
    self.assertTrue(res_start.dispatch_pre)
    self.assertFalse(res_start.dispatch_post)
    self.assertEqual(res_start.step.id, "root-1:0")
    self.assertEqual(res_start.step.source, types.StepSource.USER)
    self.assertEqual(res_start.step.target, types.StepTarget.USER)
    self.assertEqual(res_start.step.type, types.StepType.TEXT_RESPONSE)
    self.assertEqual(res_start.step.status, types.StepStatus.ACTIVE)
    self.assertEqual(res_start.step.content, "Hello agent")
    self.assertEqual(res_start.step.content_delta, "Hello agent")
    self.assertFalse(res_start.step.is_complete_response)

    res_stop = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 0,
    })
    self.assertIsNotNone(res_stop.step)
    self.assertTrue(res_stop.dispatch_post)
    self.assertEqual(res_stop.step.status, types.StepStatus.DONE)
    self.assertEqual(res_stop.step.content, "Hello agent")
    self.assertEqual(res_stop.step.content_delta, "")
    self.assertFalse(res_stop.step.is_complete_response)

  def test_thought_and_model_output_streaming(self):
    assembler = interactions_step_assembler.InteractionsStepAssembler(
        root_interaction_id="root-1"
    )
    # 1. Thought step (index 1)
    t_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 1,
        "step": {"type": "thought"},
    })
    self.assertEqual(t_start.step.type, types.StepType.THINKING)
    self.assertEqual(t_start.step.status, types.StepStatus.ACTIVE)
    self.assertTrue(t_start.dispatch_pre)

    t_delta1 = assembler.handle_step_delta({
        "event_type": "step.delta",
        "index": 1,
        "delta": {
            "type": "raw_thought",
            "content": {"type": "text", "text": "Thinking "},
        },
    })
    self.assertEqual(t_delta1.step.thinking, "Thinking ")
    self.assertEqual(t_delta1.step.thinking_delta, "Thinking ")

    t_delta2 = assembler.handle_step_delta({
        "event_type": "step.delta",
        "index": 1,
        "delta": {
            "type": "raw_thought",
            "content": {"type": "text", "text": "deeply..."},
        },
    })
    self.assertEqual(t_delta2.step.thinking, "Thinking deeply...")
    self.assertEqual(t_delta2.step.thinking_delta, "deeply...")

    t_stop = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 1,
    })
    self.assertEqual(t_stop.step.status, types.StepStatus.DONE)
    self.assertEqual(t_stop.step.thinking, "Thinking deeply...")
    self.assertEqual(t_stop.step.thinking_delta, "")
    self.assertTrue(t_stop.dispatch_post)

    # 2. Model output step (index 2)
    m_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 2,
        "step": {"type": "model_output"},
    })
    self.assertEqual(m_start.step.type, types.StepType.TEXT_RESPONSE)
    self.assertFalse(m_start.step.is_complete_response)

    m_delta1 = assembler.handle_step_delta({
        "event_type": "step.delta",
        "index": 2,
        "delta": {"type": "text", "text": "Hello "},
    })
    self.assertEqual(m_delta1.step.content, "Hello ")
    self.assertEqual(m_delta1.step.content_delta, "Hello ")
    self.assertFalse(m_delta1.step.is_complete_response)

    m_delta2 = assembler.handle_step_delta({
        "event_type": "step.delta",
        "index": 2,
        "delta": {"type": "text", "text": "world!"},
    })
    self.assertEqual(m_delta2.step.content, "Hello world!")
    self.assertEqual(m_delta2.step.content_delta, "world!")

    m_stop = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 2,
        "usage": {
            "total_input_tokens": 100,
            "total_cached_tokens": 20,
            "total_output_tokens": 30,
            "total_thought_tokens": 10,
            "total_tokens": 140,
        },
    })
    self.assertEqual(m_stop.step.status, types.StepStatus.DONE)
    self.assertEqual(m_stop.step.content, "Hello world!")
    self.assertEqual(m_stop.step.content_delta, "")
    self.assertTrue(m_stop.step.is_complete_response)
    self.assertIsNotNone(m_stop.step_usage)
    self.assertEqual(m_stop.step_usage.prompt_token_count, 100)
    self.assertEqual(m_stop.step_usage.cached_content_token_count, 20)
    self.assertEqual(m_stop.step_usage.candidates_token_count, 30)
    self.assertEqual(m_stop.step_usage.thoughts_token_count, 10)
    self.assertEqual(m_stop.step_usage.total_token_count, 140)

  def test_builtin_and_custom_tool_call_result_pairing(self):
    assembler = interactions_step_assembler.InteractionsStepAssembler(
        root_interaction_id="root-1"
    )
    # 1. view_file_call at index 1, view_file_result at index 2
    vf_call_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 1,
        "step": {
            "type": "view_file_call",
            "description": "Viewing file:///tmp/test.py",
            "file_path": "file:///tmp/test.py",
            "start_line": 1,
            "end_line": 50,
        },
    })
    self.assertIsNotNone(vf_call_start.step)
    self.assertEqual(vf_call_start.step.type, types.StepType.TOOL_CALL)
    self.assertEqual(vf_call_start.step.status, types.StepStatus.ACTIVE)
    self.assertEqual(vf_call_start.step.target, types.StepTarget.ENVIRONMENT)
    self.assertLen(vf_call_start.step.tool_calls, 1)
    tc = vf_call_start.step.tool_calls[0]
    self.assertEqual(tc.name, "view_file")
    self.assertEqual(
        tc.args, {"file_path": "/tmp/test.py", "start_line": 1, "end_line": 50}
    )
    self.assertEqual(tc.canonical_path, "/tmp/test.py")

    # step.stop on the call step itself should not emit a duplicate step
    vf_call_stop = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 1,
    })
    self.assertIsNone(vf_call_stop.step)

    # view_file_result at index 2
    vf_res_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 2,
        "step": {
            "type": "view_file_result",
            "description": "Viewing file:///tmp/test.py",
        },
    })
    self.assertIsNone(vf_res_start.step)

    vf_res_stop = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 2,
    })
    self.assertIsNotNone(vf_res_stop.step)
    self.assertTrue(vf_res_stop.dispatch_post)
    self.assertEqual(vf_res_stop.step.status, types.StepStatus.DONE)
    self.assertEqual(vf_res_stop.step.tool_calls[0].id, tc.id)
    self.assertFalse(vf_res_stop.step.is_complete_response)

    # 2. function_call with error function_result
    fn_call_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 3,
        "step": {
            "type": "function_call",
            "id": "call-abc",
            "name": "custom_op",
            "arguments": {"x": 5},
        },
    })
    self.assertEqual(fn_call_start.step.tool_calls[0].id, "call-abc")
    self.assertEqual(fn_call_start.step.tool_calls[0].name, "custom_op")
    self.assertEqual(fn_call_start.step.tool_calls[0].args, {"x": 5})

    assembler.handle_step_stop({"event_type": "step.stop", "index": 3})
    assembler.handle_step_start({
        "event_type": "step.start",
        "index": 4,
        "step": {
            "type": "function_result",
            "call_id": "call-abc",
            "name": "custom_op",
            "is_error": True,
            "result": "Database offline",
        },
    })
    fn_res_stop = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 4,
    })
    self.assertEqual(fn_res_stop.step.status, types.StepStatus.ERROR)
    self.assertEqual(fn_res_stop.step.error, "Database offline")
    self.assertEqual(fn_res_stop.step.tool_calls[0].id, "call-abc")

  def test_elicitation_and_subagent_steps(self):
    assembler = interactions_step_assembler.InteractionsStepAssembler(
        root_interaction_id="root-1"
    )
    assembler.record_parent_trajectory("sub-1", "root-1")
    assembler.record_parent_trajectory("sub-2", "sub-1")

    el_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 5,
        "sub_interaction_id": "sub-2",
        "step": {
            "type": "elicitation_call",
            "description": "Prompted for questions",
            "elicitation_id": "sub-2-question-2-0",
            "preamble": {
                "content": [{"type": "text", "text": "Python or Go?"}]
            },
            "multiple_choice_request": {
                "choices": [
                    {
                        "label": "1",
                        "display": [{"type": "text", "text": "Python"}],
                    },
                    {"label": "2", "display": [{"type": "text", "text": "Go"}]},
                ],
                "allow_multiple_selections": False,
            },
        },
    })
    self.assertIsNotNone(el_start.step)
    self.assertEqual(el_start.step.trajectory_id, "sub-2")
    self.assertEqual(el_start.step.parent_trajectory_id, "sub-1")
    self.assertEqual(el_start.step.depth, 2)
    self.assertEqual(el_start.step.status, types.StepStatus.WAITING_FOR_USER)
    self.assertIsNotNone(el_start.elicitation)
    self.assertEqual(el_start.elicitation.elicitation_id, "sub-2-question-2-0")
    self.assertEqual(el_start.elicitation.group_key, "sub-2-question-2")
    self.assertEqual(el_start.elicitation.question.question, "Python or Go?")
    self.assertLen(el_start.elicitation.question.options, 2)
    self.assertEqual(el_start.elicitation.question.options[0].id, "1")
    self.assertEqual(el_start.elicitation.question.options[0].text, "Python")

    assembler.handle_step_start({
        "event_type": "step.start",
        "index": 6,
        "sub_interaction_id": "sub-2",
        "step": {
            "type": "elicitation_result",
            "elicitation_id": "sub-2-question-2-0",
            "multiple_choice": {"selected_choice_labels": ["1"]},
        },
    })
    el_res_stop = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 6,
        "sub_interaction_id": "sub-2",
    })
    self.assertIsNotNone(el_res_stop.step)
    self.assertEqual(el_res_stop.step.status, types.StepStatus.DONE)
    self.assertEqual(
        el_res_stop.step.tool_calls[0].args, {"question": "Python or Go?"}
    )

  def test_tool_call_preserves_additional_fields(self):
    assembler = interactions_step_assembler.InteractionsStepAssembler(
        root_interaction_id="root-1"
    )
    vf_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 1,
        "step": {
            "type": "view_file_call",
            "id": "call-vf-extra",
            "file_path": "file:///workspace/src/main.py",
            "start_line": 1,
            "end_line": 50,
            "content_offset": 1024,
        },
    })
    self.assertIsNotNone(vf_start.step)
    self.assertEqual(
        vf_start.step.tool_calls[0].args,
        {
            "file_path": "/workspace/src/main.py",
            "start_line": 1,
            "end_line": 50,
            "content_offset": 1024,
        },
    )

  def test_delta_before_start_and_empty_model_output_stop(self):
    assembler = interactions_step_assembler.InteractionsStepAssembler(
        root_interaction_id="root-1"
    )
    # 1. Receiving step.delta before step.start synthesizes state and sets
    # dispatch_pre=True on the first delta only.
    d1 = assembler.handle_step_delta({
        "event_type": "step.delta",
        "index": 10,
        "delta": {"type": "text", "text": "Synthesized "},
    })
    self.assertIsNotNone(d1.step)
    self.assertTrue(d1.dispatch_pre)
    self.assertEqual(d1.step.type, types.StepType.TEXT_RESPONSE)
    self.assertEqual(d1.step.content, "Synthesized ")

    d2 = assembler.handle_step_delta({
        "event_type": "step.delta",
        "index": 10,
        "delta": {"type": "text", "text": "step"},
    })
    self.assertFalse(d2.dispatch_pre)
    self.assertEqual(d2.step.content, "Synthesized step")

    stop10 = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 10,
    })
    self.assertTrue(stop10.dispatch_post)
    self.assertTrue(stop10.step.is_complete_response)

    # 2. Stopping an empty model_output step downgrades type to UNKNOWN and
    # sets is_complete_response=False.
    assembler.handle_step_start({
        "event_type": "step.start",
        "index": 11,
        "step": {"type": "model_output"},
    })
    empty_stop = assembler.handle_step_stop({
        "event_type": "step.stop",
        "index": 11,
    })
    self.assertEqual(empty_stop.step.type, types.StepType.UNKNOWN)
    self.assertFalse(empty_stop.step.is_complete_response)

  def test_additional_builtin_tool_call_and_result_subtypes(self):
    assembler = interactions_step_assembler.InteractionsStepAssembler(
        root_interaction_id="root-1"
    )
    # 1. code_execution_call / code_execution_result (error with exit_code)
    ce_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 1,
        "step": {
            "type": "code_execution_call",
            "id": "ce-1",
            "code": "ls -la",
        },
    })
    self.assertEqual(ce_start.step.tool_calls[0].name, "run_command")
    self.assertEqual(
        ce_start.step.tool_calls[0].args,
        {"command_line": "ls -la", "language": "bash"},
    )
    assembler.handle_step_stop({"event_type": "step.stop", "index": 1})
    assembler.handle_step_start({
        "event_type": "step.start",
        "index": 2,
        "step": {
            "type": "code_execution_result",
            "call_id": "ce-1",
            "is_error": True,
            "result": "",
            "exit_code": 2,
        },
    })
    ce_res = assembler.handle_step_stop({"event_type": "step.stop", "index": 2})
    self.assertEqual(ce_res.step.status, types.StepStatus.ERROR)
    self.assertEqual(ce_res.step.error, "Command failed with exit code 2")

    # 2. generate_image_call (with image_paths list normalization) /
    # generate_image_result (merging result fields into ToolCall.args)
    gi_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 3,
        "step": {
            "type": "generate_image_call",
            "id": "gi-1",
            "prompt": "A sunset",
            "image_paths": ["file:///tmp/ref1.png", "/tmp/ref2.png"],
        },
    })
    self.assertEqual(gi_start.step.tool_calls[0].name, "generate_image")
    self.assertEqual(
        gi_start.step.tool_calls[0].args,
        {
            "prompt": "A sunset",
            "image_paths": ["/tmp/ref1.png", "/tmp/ref2.png"],
        },
    )
    assembler.handle_step_stop({"event_type": "step.stop", "index": 3})
    assembler.handle_step_start({
        "event_type": "step.start",
        "index": 4,
        "step": {
            "type": "generate_image_result",
            "call_id": "gi-1",
            "image_name": "sunset_out",
            "aspect_ratio": "16:9",
        },
    })
    gi_res = assembler.handle_step_stop({"event_type": "step.stop", "index": 4})
    self.assertEqual(
        gi_res.step.tool_calls[0].args,
        {
            "prompt": "A sunset",
            "image_paths": ["/tmp/ref1.png", "/tmp/ref2.png"],
            "image_name": "sunset_out",
            "aspect_ratio": "16:9",
        },
    )

    # 3. google_search_call / google_search_result
    gs_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 5,
        "step": {
            "type": "google_search_call",
            "id": "gs-1",
            "queries": ["antigravity sdk", "gaos interactions"],
        },
    })
    self.assertEqual(gs_start.step.tool_calls[0].name, "search_web")
    self.assertEqual(
        gs_start.step.tool_calls[0].args["query"], "antigravity sdk"
    )
    assembler.handle_step_stop({"event_type": "step.stop", "index": 5})
    assembler.handle_step_start({
        "event_type": "step.start",
        "index": 6,
        "step": {
            "type": "google_search_result",
            "call_id": "gs-1",
            "is_error": True,
            "result": [{"search_suggestions": "Try another query"}],
        },
    })
    gs_res = assembler.handle_step_stop({"event_type": "step.stop", "index": 6})
    self.assertEqual(gs_res.step.status, types.StepStatus.ERROR)
    self.assertEqual(gs_res.step.error, "Try another query")

    # 4. url_context_call / url_context_result & skill_lookup_call / result
    uc_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 7,
        "step": {
            "type": "url_context_call",
            "id": "uc-1",
            "urls": ["https://example.com"],
        },
    })
    self.assertEqual(uc_start.step.tool_calls[0].name, "read_url_content")
    self.assertEqual(
        uc_start.step.tool_calls[0].args["url"], "https://example.com"
    )
    assembler.handle_step_stop({"event_type": "step.stop", "index": 7})
    assembler.handle_step_start({
        "event_type": "step.start",
        "index": 8,
        "step": {
            "type": "url_context_result",
            "call_id": "uc-1",
            "is_error": True,
        },
    })
    uc_res = assembler.handle_step_stop({"event_type": "step.stop", "index": 8})
    self.assertEqual(uc_res.step.error, "Reading URL content failed")

    sl_start = assembler.handle_step_start({
        "event_type": "step.start",
        "index": 9,
        "step": {
            "type": "skill_lookup_call",
            "id": "sl-1",
            "requested_skill_names": ["critique"],
        },
    })
    self.assertEqual(
        sl_start.step.tool_calls[0].args,
        {"operation": "", "requested_skill_names": ["critique"]},
    )
    assembler.handle_step_stop({"event_type": "step.stop", "index": 9})
    assembler.handle_step_start({
        "event_type": "step.start",
        "index": 10,
        "step": {
            "type": "skill_lookup_result",
            "call_id": "sl-1",
            "error_message": "Skill lookup failed",
        },
    })
    sl_res = assembler.handle_step_stop(
        {"event_type": "step.stop", "index": 10}
    )
    self.assertEqual(sl_res.step.error, "Skill lookup failed")


if __name__ == "__main__":
  absltest.main()
