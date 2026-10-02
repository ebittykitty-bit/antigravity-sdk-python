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

"""Unit and integration tests for interactions_connection."""

import asyncio
import json
import unittest
from unittest import mock

from absl.testing import absltest

from google.antigravity import types
from google.antigravity.connections.local import interactions_connection
from google.antigravity.connections.local import local_connection_config
from google.antigravity.connections.local import test_utils
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


class InteractionsConnectionUnitTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)
    self.mock_process = mock.MagicMock()
    self.ws = test_utils.TestWebSocket()

  async def test_send_cancel_and_disconnect_with_session_end_hook(self):
    ended = asyncio.Event()

    @hooks.on_session_end
    async def handle_end():
      ended.set()

    hr = hook_runner.HookRunner(on_session_end_hooks=[handle_end])

    conn = interactions_connection.InteractionsConnection(
        process=self.mock_process,
        ws=self.ws,
        root_interaction_id="conv-root",
        hook_runner=hr,
    )
    self.assertEqual(conn.conversation_id, "conv-root")

    # 1. send()
    await conn.send("Hello from Interactions")
    raw_sent = await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        json.loads(raw_sent),
        {
            "event_type": "input",
            "content": [{"type": "text", "text": "Hello from Interactions"}],
        },
    )

    # 2. cancel()
    await conn.cancel()
    raw_cancel = await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        json.loads(raw_cancel), {"event_type": "interaction.cancel"}
    )

    # 3. disconnect() with OnSessionEnd hook
    disconnect_task = asyncio.create_task(conn.disconnect())
    raw_complete = await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        json.loads(raw_complete), {"event_type": "interaction.complete"}
    )

    # Simulate harness sending call_hook_request(on_session_end) then
    # interaction.completed.
    await self.ws.queue.put(
        json.dumps({
            "event_type": "call_hook_request",
            "request_id": "req-end-1",
            "type": "on_session_end",
            "name": "OnSessionEnd",
        })
    )
    raw_hook_resp = await asyncio.wait_for(
        self.ws.sent_queue.get(), timeout=2.0
    )
    self.assertEqual(
        json.loads(raw_hook_resp),
        {
            "event_type": "call_hook_response",
            "request_id": "req-end-1",
            "empty_result": {},
        },
    )
    await self.ws.queue.put(json.dumps({"event_type": "interaction.completed"}))
    await asyncio.wait_for(disconnect_task, timeout=2.0)
    self.assertTrue(ended.is_set())

  def test_interactions_agent_config_creates_interactions_strategy(self):
    cfg = local_connection_config.InteractionsAgentConfig(
        api_key="test-key",
        system_instructions="You are helpful.",
    )
    strategy = cfg.create_strategy(tool_runner=None, hook_runner=None)
    self.assertIsInstance(
        strategy, interactions_connection.InteractionsConnectionStrategy
    )
    create_ev = strategy._build_create_interaction_event()
    self.assertEqual(create_ev["event_type"], "interaction.create")
    self.assertEqual(create_ev["agent"], "antigravity")
    self.assertEqual(create_ev["agent_config"]["type"], "antigravity")

  def test_rejects_auto_and_dynamic_policies(self):
    for unsupported_policy in (
        policy.auto(),
        policy.allow("run_command", when=lambda args: True),
    ):
      cfg = local_connection_config.InteractionsAgentConfig(
          api_key="test-key",
          policies=[unsupported_policy],
      )
      strategy = cfg.create_strategy(tool_runner=None, hook_runner=None)
      with self.assertRaises(types.AntigravityValidationError):
        strategy._build_create_interaction_event()

  async def test_aenter_handshake_failure_kills_process(self):
    cfg = local_connection_config.InteractionsAgentConfig(
        api_key="test-key",
    )
    strategy = cfg.create_strategy(tool_runner=None, hook_runner=None)
    self.mock_process.stderr.read.return_value = b"fatal handshake error"
    await self.ws.queue.put(
        json.dumps({"event_type": "error", "error": {"message": "bad config"}})
    )
    with mock.patch.object(
        strategy,
        "_spawn_harness_and_connect_ws",
        new=mock.AsyncMock(
            return_value=(self.mock_process, self.ws, "ws://127.0.0.1:9999")
        ),
    ):
      with self.assertRaisesRegex(RuntimeError, "fatal handshake error"):
        await strategy.__aenter__()
    self.mock_process.kill.assert_called_once()

  async def test_rejects_triggers(self):
    from google.antigravity.triggers import triggers  # pylint: disable=g-import-not-at-top

    @triggers.trigger
    async def my_trigger(ctx: triggers.TriggerContext) -> None:
      await ctx.send("ping")

    with self.assertRaises(types.AntigravityValidationError):
      local_connection_config.InteractionsAgentConfig(
          api_key="test-key",
          triggers=[my_trigger],
      )

    conn = interactions_connection.InteractionsConnection(
        process=self.mock_process,
        ws=self.ws,
        root_interaction_id="conv-root",
    )
    with self.assertRaises(NotImplementedError):
      await conn.send_trigger_notification("ping")


if __name__ == "__main__":
  absltest.main()
