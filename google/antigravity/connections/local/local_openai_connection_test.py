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

"""Unit tests for LocalOpenAIConnectionStrategy and LocalOpenAIAgentConfig."""

import unittest
from unittest import mock

from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import local_openai_connection
from google.antigravity.connections.local import local_openai_connection_config
from google.antigravity.connections.local import test_utils
from google.antigravity.hooks import policy
from google.antigravity.tools import tool_runner as tool_runner_mod


class LocalOpenAIConnectionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)

  def test_local_openai_strategy_harness_config(self):
    """Verify generic external OpenAI configuration works and clears Gemini config."""
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )

    self.assertIsInstance(
        strategy, local_openai_connection.LocalOpenAIConnectionStrategy
    )
    h_cfg = strategy._build_harness_config()

    # pylint: disable-next=g-generic-assert
    self.assertEqual(len(h_cfg.models), 1)
    model = h_cfg.models[0]
    self.assertEqual(model.name, "llama3.1")
    self.assertEqual(model.types, [localharness_pb2.MODEL_TYPE_TEXT])
    self.assertTrue(model.HasField("gemma_endpoint"))
    self.assertEqual(model.gemma_endpoint.base_url, "http://localhost:11434/v1")

  def test_local_openai_strategy_validate_empty_base_url(self):
    """Verify LocalOpenAIConnectionStrategy validates non-empty base_url."""
    strategy = local_openai_connection.LocalOpenAIConnectionStrategy(
        base_url="",
        model_name="test",
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    with self.assertRaises(types.AntigravityValidationError):
      strategy._validate_connection()

  def test_local_openai_config_model_target_parsing(self):
    """Verify LocalOpenAIAgentConfig parses model and endpoint base_url from ModelTarget."""
    endpoint = types.GeminiAPIEndpoint(base_url="http://custom-ollama:11434/v1")
    target = types.ModelTarget(name="llama3.2", endpoint=endpoint)
    config = local_openai_connection_config.LocalOpenAIAgentConfig(model=target)
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._model_name, "llama3.2")
    self.assertEqual(strategy._base_url, "http://custom-ollama:11434/v1")

  def test_local_openai_config_default_capabilities(self):
    """Verify LocalOpenAIAgentConfig defaults to all capabilities enabled."""
    openai_config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    )
    self.assertIsNone(openai_config.capabilities.enabled_tools)
    self.assertIsNone(openai_config.capabilities.disabled_tools)

  def test_local_openai_config_lightweight_method(self):
    """Verify LocalOpenAIAgentConfig.lightweight returns LocalOpenAIAgentConfig with defaults."""
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    ).lightweight()
    self.assertIsInstance(
        config, local_openai_connection_config.LocalOpenAIAgentConfig
    )
    self.assertEqual(config.base_url, "http://localhost:11434/v1")
    self.assertEqual(config.model, "llama3.1")
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertIsNotNone(config.compaction_config)
    self.assertEqual(config.compaction_config.token_threshold, 65536)
    self.assertIsNone(config.capabilities.compaction_threshold)
    self.assertFalse(config.capabilities.enable_subagents)

  def test_local_openai_config_lightweight_method_preserves_explicit_compaction(
      self,
  ):
    """Verify LocalOpenAIAgentConfig.lightweight preserves explicit compaction_config."""
    user_compaction = types.CompactionConfig(
        token_threshold=20000,
    )
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        compaction_config=user_compaction,
    ).lightweight()
    self.assertEqual(config.compaction_config.token_threshold, 20000)

  def test_local_openai_config_lightweight_method_with_overrides(self):
    """Verify LocalOpenAIAgentConfig.lightweight respects capability overrides."""
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        capabilities=types.CapabilitiesConfig(compaction_threshold=4000),
    ).lightweight()
    self.assertIsInstance(
        config, local_openai_connection_config.LocalOpenAIAgentConfig
    )
    self.assertEqual(config.capabilities.compaction_threshold, 4000)
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertFalse(config.capabilities.enable_subagents)

  def test_local_openai_config_workspace_policies(self):
    """Verify LocalOpenAIAgentConfig does not prepend workspace_only policy."""
    config_openai = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost",
        model="m",
        workspaces=["/tmp/my_workspace"],
    )
    # LocalOpenAI uses localharness for native workspace containment;
    # config_openai.policies contains only the 2 confirm_run_command policies.
    # pylint: disable-next=g-generic-assert
    self.assertEqual(len(config_openai.policies), 2)
    self.assertEqual(config_openai.policies[0].tool, "run_command")
    self.assertEqual(config_openai.policies[1].tool, "*")

  def test_local_openai_config_mcp_servers_and_subagents_passed_to_strategy(
      self,
  ):
    """Verify LocalOpenAIAgentConfig passes mcp_servers and subagents to strategy."""
    mcp_server = types.McpStdioServer(
        name="test_mcp", command="echo", args=["hello"]
    )
    subagent = types.SubagentConfig(
        name="test_subagent",
        description="A test subagent",
        system_instructions="You are a subagent",
    )
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        mcp_servers=[mcp_server],
        subagents=[subagent],
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._mcp_servers, [mcp_server])
    self.assertEqual(strategy._subagents, [subagent])

  def test_policies_forwarded_to_strategy_and_harness_config(self):
    """Verify explicit policies are forwarded to LocalOpenAIConnectionStrategy."""
    denies = [
        policy.deny("create_file"),
        policy.deny("edit_file"),
        policy.deny("run_command"),
    ]
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        policies=denies,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._policies, denies)
    h_cfg = strategy._build_harness_config()
    self.assertTrue(h_cfg.HasField("policy_config"))
    self.assertEqual(len(h_cfg.policy_config.rules), 3)

  def test_default_policies_forwarded_to_strategy(self):
    """Verify default confirm_run_command() policy reaches strategy when policies is omitted."""
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(len(strategy._policies), 2)
    self.assertEqual(strategy._policies[0].tool, "run_command")
    self.assertEqual(strategy._policies[0].decision, policy.Decision.DENY)
    h_cfg = strategy._build_harness_config()
    self.assertTrue(h_cfg.HasField("policy_config"))
    self.assertEqual(len(h_cfg.policy_config.rules), 2)

  def test_budget_and_session_continuation_mode_forwarded_to_strategy(self):
    """Verify budget_config and session_continuation_mode are forwarded."""
    budget = types.BudgetConfig(max_model_calls=7, max_total_tokens=12000)
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        conversation_id="a" * 32,
        session_continuation_mode=types.SessionContinuationMode.RESUME,
        budget_config=budget,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(
        strategy._session_continuation_mode,
        types.SessionContinuationMode.RESUME,
    )
    self.assertEqual(strategy._budget_config, budget)
    h_cfg = strategy._build_harness_config()
    self.assertEqual(
        h_cfg.session_continuation_mode,
        localharness_pb2.HarnessConfig.RESUME,
    )
    self.assertTrue(h_cfg.HasField("budget_config"))
    self.assertEqual(h_cfg.budget_config.max_model_calls, 7)
    self.assertEqual(h_cfg.budget_config.max_total_tokens, 12000)

  def test_tools_forwarded_to_strategy(self):
    """Verify custom tools are forwarded to strategy and included in harness proto."""

    def custom_lookup(query: str) -> str:
      """Looks up a query."""
      return query

    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        tools=[custom_lookup],
    )
    t_runner = tool_runner_mod.ToolRunner(tools=config._get_all_custom_tools())
    strategy = config.create_strategy(
        tool_runner=t_runner,
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._tools, [custom_lookup])
    h_cfg = strategy._build_harness_config()
    tool_names = [t.name for t in h_cfg.tools]
    self.assertIn("custom_lookup", tool_names)


if __name__ == "__main__":
  unittest.main()
