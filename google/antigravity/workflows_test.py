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

from absl.testing import absltest
from google.antigravity import beta
from google.antigravity import workflows


class WorkflowsTest(absltest.TestCase):

  def test_beta_workflows_export_and_qualifier_stripping(self):
    self.assertIs(beta.workflows, workflows)
    self.assertIn("workflows", beta.__all__)

    @beta.workflows.define
    async def beta_qualified_pipeline():
      """Pipeline using beta.workflows.<primitive>."""
      beta.workflows.phase("Discover")
      res = await beta.workflows.agent(
          "Check status", schema={"type": "string"}
      )
      beta.workflows.log(res)

    self.assertIsInstance(beta_qualified_pipeline, workflows.WorkflowDefinition)
    self.assertNotIn("beta.workflows", beta_qualified_pipeline.source)
    self.assertIn("phase('Discover')", beta_qualified_pipeline.source)
    self.assertIn(
        "res = await agent('Check status', schema={'type': 'string'})",
        beta_qualified_pipeline.source,
    )
    self.assertIn("log(res)", beta_qualified_pipeline.source)

  def test_define_with_qualified_primitives(self):
    @workflows.define
    async def audit_pipeline():
      """Audit modules in parallel."""
      workflows.phase("Discover")
      workflows.log("Starting discovery")
      modules = await workflows.agent(
          "List modules",
          schema={"type": "array", "items": {"type": "string"}},
      )
      workflows.phase("Analyze")
      results = await workflows.parallel(
          modules,
          lambda item: workflows.agent(
              f"Analyze {item}",
              schema={"type": "string"},
              role="Analyzer",
          ),
      )
      summary = await workflows.pipeline(
          results,
          lambda item: workflows.agent(
              f"Summarize {item}",
              schema={"type": "string"},
          ),
      )
      workflows.log(summary)

    self.assertIsInstance(audit_pipeline, workflows.WorkflowDefinition)
    self.assertEqual(audit_pipeline.name, "audit_pipeline")
    self.assertEqual(audit_pipeline.description, "Audit modules in parallel.")
    self.assertNotIn("workflows.phase", audit_pipeline.source)
    self.assertIn("phase('Discover')", audit_pipeline.source)
    self.assertIn("await parallel(modules,", audit_pipeline.source)
    self.assertIn("await pipeline(results,", audit_pipeline.source)

  def test_define_allows_nested_async_helper_with_return(self):
    @workflows.define
    async def pipeline_with_helper():
      async def fix(bug: str):
        test = await workflows.agent(
            f"Write test for {bug}",
            schema={"type": "string"},
            role=f"Test {bug}",
        )
        return await workflows.agent(
            f"Fix {bug} using {test}",
            schema={"type": "string"},
            role=f"Fix {bug}",
        )

      fixes = await workflows.pipeline(["BUG-1.md", "BUG-2.md"], fix)
      for f in fixes:
        workflows.log("fixed:", f)

    self.assertIn("async def fix(", pipeline_with_helper.source)
    self.assertIn("return await agent(", pipeline_with_helper.source)
    self.assertIn(
        "fixes = await pipeline(['BUG-1.md', 'BUG-2.md'], fix)",
        pipeline_with_helper.source,
    )

  def test_define_with_unqualified_primitives(self):
    phase = workflows.phase
    log = workflows.log
    agent = workflows.agent

    @workflows.define
    async def simple_pipeline():
      phase("Step 1")
      res = await agent("Run step 1", schema={"type": "string"})
      log("Done:", res)

    self.assertEqual(simple_pipeline.name, "simple_pipeline")
    self.assertEqual(simple_pipeline.description, "")
    self.assertIn("phase('Step 1')", simple_pipeline.source)
    self.assertIn(
        "res = await agent('Run step 1', schema={'type': 'string'})",
        simple_pipeline.source,
    )
    self.assertIn("log('Done:', res)", simple_pipeline.source)

  def test_define_preserves_multiline_strings_and_one_liner(self):
    phase = workflows.phase
    log = workflows.log

    @workflows.define
    def multiline_wf():
      prompt = """Line 1
Line 2"""
      log(prompt)

    ns: dict[str, object] = {}
    captured: list[str] = []
    ns["log"] = captured.append
    exec(multiline_wf.source, ns)  # pylint: disable=exec-used
    self.assertEqual(captured, ["Line 1\nLine 2"])

    @workflows.define
    def one_liner():
      phase("SingleLine")  # pylint: disable=multiple-statements

    self.assertEqual(one_liner.source.strip(), "phase('SingleLine')")

  def test_validate_rejects_return_and_yield(self):
    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "return/yield statements are not allowed in workflow scripts",
    ):

      @workflows.define
      def with_return():
        workflows.phase("Step")
        return "done"

      del with_return

    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "return/yield statements are not allowed in workflow scripts",
    ):
      workflows.validate_workflow_source("yield 'x'\n")

    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "return/yield statements are not allowed in workflow scripts",
    ):
      workflows.validate_workflow_source("def gen():\n  yield 1\n")

    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "return/yield statements are not allowed in workflow scripts",
    ):
      workflows.validate_workflow_source("def gen_from():\n  yield from [1]\n")

  def test_validate_rejects_import(self):
    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "import statements are not allowed in workflow scripts",
    ):
      workflows.validate_workflow_source("import os\nphase('test')\n")

    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "import statements are not allowed in workflow scripts",
    ):
      workflows.validate_workflow_source("from os import path\n")

  def test_validate_rejects_while_loop(self):
    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "while loops are not allowed in workflow scripts",
    ):

      @workflows.define
      def bad_loop():
        while True:
          workflows.log("loop")

      del bad_loop

  def test_validate_rejects_private_attribute(self):
    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "attribute '_secret' is not allowed in workflow scripts",
    ):
      workflows.validate_workflow_source("x = obj._secret\n")

  def test_validate_rejects_dunder_name(self):
    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "name '__import__' is not allowed in workflow scripts",
    ):
      workflows.validate_workflow_source("__import__('os')\n")

  def test_validate_rejects_syntax_error(self):
    with self.assertRaisesRegex(
        workflows.WorkflowError, "workflow syntax error"
    ):
      workflows.validate_workflow_source("def broken(:\n")

  def test_define_rejects_required_arguments(self):
    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "workflow functions cannot require positional/keyword arguments",
    ):

      @workflows.define  # type: ignore[arg-type]
      def parameterized_wf(target: str):
        workflows.phase(target)

      del parameterized_wf

    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "workflow functions cannot require positional/keyword arguments",
    ):

      @workflows.define  # type: ignore[arg-type]
      def wf_with_default(branch: str = "main"):
        workflows.phase(branch)

      del wf_with_default

    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "workflow functions cannot require positional/keyword arguments",
    ):

      @workflows.define  # type: ignore[arg-type]
      def wf_with_varargs(*args, **kwargs):
        del args, kwargs
        workflows.phase("Step")

      del wf_with_varargs

  def test_define_rejects_docstring_only_function(self):
    with self.assertRaisesRegex(
        workflows.WorkflowError,
        "workflow function body cannot be empty",
    ):

      @workflows.define
      def docstring_only_wf():
        """Only a docstring, no statements."""

      del docstring_only_wf

  def test_primitives_raise_outside_runtime(self):
    with self.assertRaises(TypeError):
      workflows.log()
    with self.assertRaisesRegex(RuntimeError, "workflows.log"):
      workflows.log("hello")
    with self.assertRaisesRegex(RuntimeError, "workflows.phase"):
      workflows.phase("Explore")
    with self.assertRaisesRegex(RuntimeError, "workflows.agent"):
      workflows.agent("Prompt", schema={"type": "string"})
    with self.assertRaisesRegex(RuntimeError, "workflows.parallel"):
      workflows.parallel(
          ["a"], lambda x: workflows.agent(x, schema={"type": "string"})
      )
    with self.assertRaisesRegex(RuntimeError, "workflows.pipeline"):
      workflows.pipeline(
          ["a"], lambda x: workflows.agent(x, schema={"type": "string"})
      )


if __name__ == "__main__":
  absltest.main()
