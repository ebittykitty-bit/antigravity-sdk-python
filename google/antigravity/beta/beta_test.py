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

"""Unit tests for the Google Antigravity SDK `beta` namespace and `@beta` decorator."""

import asyncio
import copy
from unittest import mock

from absl.testing import absltest

from google.antigravity import agent as agent_lib
from google.antigravity import beta as beta_pkg
from google.antigravity.beta import beta
from google.antigravity.beta import BetaNamespace
from google.antigravity.beta import is_beta


class BetaNamespaceTest(absltest.TestCase):
  """Tests for `@beta` module exports and `.beta` class/instance method binding."""

  def setUp(self):
    super().setUp()
    self._initial_all = list(beta_pkg._ALL_EXPORTS)
    self._initial_exports = dict(beta_pkg._BETA_EXPORTS)
    self._initial_lazy_loaded = beta_pkg._LAZY_SUBMODULES_LOADED

  def tearDown(self):
    for key in list(beta_pkg._BETA_EXPORTS.keys()):
      if key not in self._initial_exports:
        beta_pkg._BETA_EXPORTS.pop(key, None)
        beta_pkg.__dict__.pop(key, None)
    beta_pkg._ALL_EXPORTS[:] = self._initial_all
    beta_pkg.__dict__["__all__"] = beta_pkg._ALL_EXPORTS
    beta_pkg._LAZY_SUBMODULES_LOADED = self._initial_lazy_loaded
    super().tearDown()

  def test_top_level_class_and_function_export_to_beta_and_all(self):
    """Verifies `@beta` exports top-level classes/functions to `beta` and `beta.__all__`."""

    @beta
    class ExperimentalToolConfig:

      def __init__(self, timeout: int = 10):
        self.timeout = timeout

    @beta
    def experimental_helper(x: int) -> int:
      return x * 2

    self.assertTrue(is_beta(ExperimentalToolConfig))
    self.assertTrue(is_beta(experimental_helper))
    self.assertIn("ExperimentalToolConfig", beta_pkg.__all__)
    self.assertIn("experimental_helper", beta_pkg.__all__)
    self.assertIs(beta_pkg.ExperimentalToolConfig, ExperimentalToolConfig)
    self.assertIs(beta_pkg.experimental_helper, experimental_helper)

    cfg = beta_pkg.ExperimentalToolConfig(timeout=25)
    self.assertEqual(cfg.timeout, 25)
    self.assertEqual(beta_pkg.experimental_helper(21), 42)

  def test_top_level_custom_export_name_and_empty_parens(self):
    """Verifies `@beta('CustomName')` and `@beta()` syntax on top-level symbols."""

    @beta("PublicBetaWidget")
    class _InternalBetaWidget:
      value = "widget"

    @beta()
    def parens_helper() -> str:
      return "ok"

    self.assertIn("PublicBetaWidget", beta_pkg.__all__)
    self.assertNotIn("_InternalBetaWidget", beta_pkg.__all__)
    self.assertIs(beta_pkg.PublicBetaWidget, _InternalBetaWidget)
    self.assertIn("parens_helper", beta_pkg.__all__)
    self.assertEqual(beta_pkg.parens_helper(), "ok")

  def test_module_pep562_getattr_and_dir_lazy_discovery(self):
    """Verifies PEP 562 `__getattr__` and `__dir__` on `beta` module."""
    beta_pkg.__dict__.pop("__all__", None)
    beta_pkg._LAZY_SUBMODULES_LOADED = False

    all_list = beta_pkg.__all__
    self.assertIn("BetaNamespace", all_list)
    self.assertTrue(beta_pkg._LAZY_SUBMODULES_LOADED)

    sentinel = object()
    beta_pkg._LAZY_SUBMODULES_LOADED = False

    def _fake_load():
      beta_pkg._LAZY_SUBMODULES_LOADED = True
      beta_pkg._BETA_EXPORTS["LazySentinelSymbol"] = sentinel
      beta_pkg._ALL_EXPORTS.append("LazySentinelSymbol")

    with mock.patch.object(
        beta_pkg, "_ensure_submodules_loaded", side_effect=_fake_load
    ):
      self.assertIs(beta_pkg.LazySentinelSymbol, sentinel)
    self.assertIn("LazySentinelSymbol", dir(beta_pkg))

    with self.assertRaises(AttributeError):
      _ = beta_pkg.definitely_non_existent_beta_symbol

  def test_owned_methods_and_properties_on_class_and_instance(self):
    """Verifies `@beta` moves owned methods/properties onto `.beta` and removes from root class."""

    class SampleOwner:

      def __init__(self, prefix: str):
        self.prefix = prefix

      def stable_method(self) -> str:
        return f"{self.prefix}:stable"

      @beta
      def experimental_sync(self, suffix: str) -> str:
        return f"{self.prefix}:{suffix}"

      @beta
      async def experimental_async(self, count: int) -> int:
        return count + len(self.prefix)

      @beta
      @classmethod
      def experimental_cm(cls, label: str) -> str:
        return f"{cls.__name__}:{label}"

      @beta
      @staticmethod
      def experimental_sm(a: int, b: int) -> int:
        return a + b

      @beta
      @property
      def experimental_prop(self) -> str:
        return f"{self.prefix}:prop"

      @beta("aliased_beta_method")
      def _internal_beta_method(self) -> str:
        return f"{self.prefix}:aliased"

      @beta()
      def empty_parens_method(self) -> str:
        return f"{self.prefix}:parens"

    owner = SampleOwner("agy")

    # 1. Stable method remains on root class/instance and is not marked beta.
    self.assertEqual(owner.stable_method(), "agy:stable")
    self.assertFalse(is_beta(SampleOwner.stable_method))
    self.assertTrue(is_beta(SampleOwner._beta_members["experimental_cm"]))
    self.assertTrue(is_beta(SampleOwner._beta_members["experimental_sm"]))
    self.assertTrue(is_beta(SampleOwner._beta_members["experimental_prop"]))

    # 2. Beta methods are NOT on the root class or instance namespace.
    self.assertFalse(hasattr(owner, "experimental_sync"))
    self.assertFalse(hasattr(SampleOwner, "experimental_sync"))
    self.assertFalse(hasattr(owner, "experimental_prop"))
    self.assertFalse(hasattr(owner, "_internal_beta_method"))
    self.assertNotIn("experimental_sync", beta_pkg.__all__)

    # 3. Instance-bound `.beta` access works for sync, async, classmethod,
    # staticmethod, custom names, and @property.
    self.assertEqual(owner.beta.experimental_sync("beta_call"), "agy:beta_call")
    self.assertEqual(
        asyncio.run(owner.beta.experimental_async(5)),
        8,
    )
    self.assertEqual(owner.beta.experimental_cm("v1"), "SampleOwner:v1")
    self.assertEqual(owner.beta.experimental_sm(3, 4), 7)
    self.assertEqual(owner.beta.experimental_prop, "agy:prop")
    self.assertEqual(owner.beta.aliased_beta_method(), "agy:aliased")
    self.assertEqual(owner.beta.empty_parens_method(), "agy:parens")

    # Shallow copy (`copy.copy`) creates an independent `.beta` bound to copy.
    owner_copy = copy.copy(owner)
    owner_copy.prefix = "copied"
    self.assertEqual(owner_copy.beta.experimental_sync("call"), "copied:call")
    self.assertEqual(owner.beta.experimental_sync("call"), "agy:call")

    # 4. Class-level `.beta` access works.
    self.assertEqual(
        SampleOwner.beta.experimental_cm("cls_call"),
        "SampleOwner:cls_call",
    )
    self.assertEqual(SampleOwner.beta.experimental_sm(10, 20), 30)
    self.assertEqual(
        SampleOwner.beta.experimental_sync(owner, "unbound"),
        "agy:unbound",
    )

    # 5. Introspection (dir and repr) on `.beta`.
    expected_members = {
        "aliased_beta_method",
        "empty_parens_method",
        "experimental_async",
        "experimental_cm",
        "experimental_prop",
        "experimental_sm",
        "experimental_sync",
    }
    self.assertEqual(set(dir(owner.beta)), expected_members)
    self.assertEqual(set(dir(SampleOwner.beta)), expected_members)
    self.assertIn("SampleOwner", repr(owner.beta))
    self.assertIn("SampleOwner", repr(SampleOwner.beta))

    with self.assertRaises(AttributeError):
      _ = owner.beta.non_existent_method

  def test_inheritance_and_mock_patch_on_beta_namespace(self):
    """Verifies subclass inheritance and `mock.patch.object` on `.beta`."""

    class BaseComponent:
      beta = BetaNamespace()

      @beta
      def base_action(self) -> str:
        return "base"

      @beta
      def overridden_action(self) -> str:
        return "base_overridden"

      @beta
      @classmethod
      def base_cm(cls) -> str:
        return f"cm:{cls.__name__}"

    class DerivedComponent(BaseComponent):

      @beta
      def derived_action(self) -> str:
        return "derived"

      @beta
      def overridden_action(self) -> str:
        return "derived_overridden"

    comp = DerivedComponent()
    self.assertEqual(comp.beta.base_action(), "base")
    self.assertEqual(comp.beta.derived_action(), "derived")
    self.assertEqual(comp.beta.overridden_action(), "derived_overridden")
    self.assertEqual(
        BaseComponent().beta.overridden_action(), "base_overridden"
    )
    self.assertEqual(DerivedComponent.beta.base_cm(), "cm:DerivedComponent")

    # 1. Class-level mock.patch.object on @classmethod.
    with mock.patch.object(
        BaseComponent.beta, "base_cm", return_value="mocked_cm"
    ):
      self.assertEqual(DerivedComponent.beta.base_cm(), "mocked_cm")
    self.assertEqual(DerivedComponent.beta.base_cm(), "cm:DerivedComponent")

    # 2. Class-level mock.patch.object on inherited method via
    # DerivedComponent.beta.
    with mock.patch.object(
        DerivedComponent.beta, "base_action", return_value="mocked_derived_base"
    ):
      self.assertEqual(comp.beta.base_action(), "mocked_derived_base")
    self.assertEqual(comp.beta.base_action(), "base")

    # 3. Instance-level mock.patch.object.
    with mock.patch.object(
        comp.beta, "base_action", return_value="mocked_inst"
    ):
      self.assertEqual(comp.beta.base_action(), "mocked_inst")
    self.assertEqual(comp.beta.base_action(), "base")

  def test_agent_has_beta_namespace(self):
    """Verifies `Agent.beta` and `agent.beta` are available on the SDK's `Agent` class."""
    self.assertIsNotNone(agent_lib.Agent.beta)
    ag = object.__new__(agent_lib.Agent)
    self.assertIsNotNone(ag.beta)
    self.assertIs(ag.beta, ag.beta)  # Cached per instance


if __name__ == "__main__":
  absltest.main()
