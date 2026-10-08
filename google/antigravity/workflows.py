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

"""Authoring utilities and typed stubs for `run_workflow` scripts."""

from __future__ import annotations

import ast
from collections.abc import Awaitable, Callable, Coroutine, Iterable
import dataclasses
import inspect
import sys
from typing import Any, Literal

from google.antigravity import beta as _beta_lib

WorkspaceMode = Literal["inherit", "branch", "share"]

_PRIMITIVE_NAMES: frozenset[str] = frozenset({
    "phase",
    "log",
    "agent",
    "parallel",
    "pipeline",
})

_WORKFLOW_MODULE_NAMES: frozenset[str] = frozenset({
    "workflows",
    "workflow",
})


class WorkflowError(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when a workflow script fails validation or runtime execution."""


# ---------------------------------------------------------------------------
# Compile-time type stubs for `run_workflow` primitives.
#
# These functions exist so that `@workflows.define` functions can be written in
# normal Python modules with full IDE autocomplete and static type-checking
# (`workflows.phase(...)`, `await workflows.parallel(...)`, etc.). At decoration
# time, `extract_workflow_source()` extracts the function body's AST and
# rewrites `workflows.<primitive>(...)` into Antigravity workflow code.
#
# Calling these stubs directly in the host SDK Python process is a user error
# and raises RuntimeError.
# ---------------------------------------------------------------------------


def _outside_runtime_error(name: str) -> RuntimeError:
  return RuntimeError(
      f"workflows.{name}() is a compile-time stub for the `run_workflow` tool"
      " and cannot be called directly in the host Python process. Call it"
      " inside a `@workflows.define` function and execute the workflow via"
      " `await agent.beta.run_workflow(workflow=...)`."
  )


def log(*args: Any) -> None:
  """Compile-time stub: emits a progress log line to the current workflow phase."""
  if not args:
    raise TypeError("log: requires at least 1 argument")
  raise _outside_runtime_error("log")


def phase(name: str) -> str:
  """Compile-time stub: starts a new named phase in the workflow execution."""
  del name
  raise _outside_runtime_error("phase")


def agent(
    prompt: str,
    schema: Any,
    role: str | None = None,
    type_name: str = "self",
    workspace: WorkspaceMode = "inherit",
) -> Coroutine[Any, Any, Any]:
  """Compile-time stub: returns an awaitable promise that runs a workflow agent."""
  del prompt, schema, role, type_name, workspace
  raise _outside_runtime_error("agent")


def parallel(
    items: Iterable[Any],
    fn: Callable[[Any], Awaitable[Any]],
) -> Coroutine[Any, Any, list[Any]]:
  """Compile-time stub: runs `await fn(item)` for each item concurrently."""
  del items, fn
  raise _outside_runtime_error("parallel")


def pipeline(
    items: Iterable[Any],
    fn: Callable[[Any], Awaitable[Any]],
) -> Coroutine[Any, Any, list[Any]]:
  """Compile-time stub: runs `await fn(item)` for each item one by one."""
  del items, fn
  raise _outside_runtime_error("pipeline")


def _is_workflow_module_ref(node: ast.AST) -> bool:
  """Returns True if `node` represents `workflows`, `workflow`, or `beta.workflows`."""
  if (
      isinstance(node, ast.Name)
      and node.id in _WORKFLOW_MODULE_NAMES
      and isinstance(node.ctx, ast.Load)
  ):
    return True
  if (
      isinstance(node, ast.Attribute)
      and isinstance(node.value, ast.Name)
      and node.value.id == "beta"
      and isinstance(node.value.ctx, ast.Load)
      and node.attr in _WORKFLOW_MODULE_NAMES
      and isinstance(node.ctx, ast.Load)
  ):
    return True
  return False


class _WorkflowQualifierStripper(ast.NodeTransformer):
  """Rewrites `(beta.)workflows.<primitive>(...)` AST nodes into Antigravity workflow code.

  When a user authors a function with `@workflows.define` or
  `@beta.workflows.define`, they write `await workflows.parallel(...)` or
  `await beta.workflows.parallel(...)` so static typecheckers resolve the
  symbol; this transformer strips the module attribute prefix when generating
  the standalone workflow script.
  """

  def __init__(self) -> None:
    super().__init__()
    self.modified = False

  def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # pylint: disable=invalid-name
    """Rewrites `(beta.)workflows.<primitive>` loads to bare `<primitive>` names."""
    self.generic_visit(node)
    if (
        _is_workflow_module_ref(node.value)
        and node.attr in _PRIMITIVE_NAMES
        and isinstance(node.ctx, ast.Load)
    ):
      self.modified = True
      return ast.copy_location(
          ast.Name(id=node.attr, ctx=ast.Load()),
          node,
      )
    return node


def _check_no_top_level_return_or_yield(nodes: list[ast.stmt]) -> None:
  """Rejects `return` at the top level of the script and `yield` anywhere."""
  stack: list[ast.AST] = list(nodes)
  while stack:
    node = stack.pop()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      # `return` is valid inside nested helper functions (such as `async def`
      # stage helpers passed to `pipeline` or `parallel`).
      for sub in ast.walk(node):
        if isinstance(sub, (ast.Yield, ast.YieldFrom)):
          raise WorkflowError(
              "return/yield statements are not allowed in workflow scripts; use"
              " workflows.log(...) to emit output"
          )
      continue
    if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
      raise WorkflowError(
          "return/yield statements are not allowed in workflow scripts; use"
          " workflows.log(...) to emit output"
      )
    stack.extend(ast.iter_child_nodes(node))


def validate_workflow_source(source: str) -> None:
  """Validates Python workflow script source against `run_workflow` AST rules.

  These checks validate the script at `@workflows.define` decoration time so
  disallowed constructs fail fast before execution:
  - `ast.Import` / `ast.ImportFrom`: workflow scripts run in a self-contained
    environment with built-in workflow primitives and do not allow imports.
  - `ast.While`: workflows must use bounded `for` loops rather than unbounded
    `while` loops.
  - `ast.Attribute` starting with `_` and `ast.Name` starting with `__`:
    private and dunder attribute/name access is disallowed in workflow scripts.
  - Top-level `ast.Return` and `ast.Yield` / `ast.YieldFrom`: disallowed at the
    top level of a workflow script (`return` inside nested `def` / `async def`
    stage helpers is permitted).

  Args:
    source: Python source code of the workflow script.

  Raises:
    WorkflowError: If the source has a syntax error or uses disallowed AST
      constructs (`import`, `while`, top-level `return`/`yield`, `_`-prefixed
      attributes, or `__`-prefixed names).
  """
  try:
    tree = ast.parse(source, "<workflow>")
  except SyntaxError as e:
    raise WorkflowError(f"workflow syntax error: {e}") from e

  _check_no_top_level_return_or_yield(tree.body)

  try:
    compile(tree, "<workflow>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
  except SyntaxError as e:
    raise WorkflowError(f"workflow syntax error: {e}") from e

  for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
      raise WorkflowError(
          "import statements are not allowed in workflow scripts"
      )
    if isinstance(node, ast.While):
      raise WorkflowError(
          "while loops are not allowed in workflow scripts; use bounded for"
          " loops"
      )
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
      raise WorkflowError(
          f"attribute {node.attr!r} is not allowed in workflow scripts"
      )
    if isinstance(node, ast.Name) and node.id.startswith("__"):
      raise WorkflowError(
          f"name {node.id!r} is not allowed in workflow scripts"
      )


def extract_workflow_source(
    fn_or_def: Callable[[], Any] | WorkflowDefinition,
) -> str:
  """Extracts and validates the standalone script source from a workflow function.

  Given a `@workflows.define` function such as:
    ```python
    @workflows.define
    async def my_workflow():
      workflows.phase("Step 1")
      res = await workflows.agent("Check status", schema={"type": "string"})
      workflows.log(res)
    ```
  this function extracts the function body, strips the docstring, and rewrites
  `(beta.)workflows.<primitive>(...)` calls into Antigravity workflow code:
    ```python
    phase('Step 1')
    res = await agent('Check status', schema={'type': 'string'})
    log(res)
    ```

  Args:
    fn_or_def: A `@workflows.define` definition or a zero-argument Python
      function (`async def` or `def`) whose body forms the workflow script.

  Returns:
    The validated Python source string ready to be written to a workflow script.

  Raises:
    WorkflowError: If the function takes arguments, its source cannot be
      extracted, or its body violates `run_workflow` AST restrictions.
  """
  if isinstance(fn_or_def, WorkflowDefinition):
    return fn_or_def.source

  if not callable(fn_or_def):
    raise TypeError(
        "Expected a callable or WorkflowDefinition, got"
        f" {type(fn_or_def).__name__}"
    )

  sig = inspect.signature(fn_or_def)
  if sig.parameters:
    raise WorkflowError(
        "workflow functions cannot require positional/keyword arguments; got"
        f" parameters {list(sig.parameters)}"
    )

  try:
    raw_source = inspect.getsource(fn_or_def)
  except (OSError, TypeError) as e:
    raise WorkflowError(
        f"cannot inspect source of workflow function {fn_or_def!r}: {e}"
    ) from e

  first_line = next(
      (line for line in raw_source.splitlines() if line.strip()), ""
  )
  is_indented = first_line[:1].isspace()
  source_to_parse = f"if True:\n{raw_source}" if is_indented else raw_source
  try:
    mod_tree = ast.parse(source_to_parse)
  except SyntaxError as e:
    raise WorkflowError(f"workflow syntax error: {e}") from e

  stmts = (
      mod_tree.body[0].body
      if is_indented and mod_tree.body and isinstance(mod_tree.body[0], ast.If)
      else mod_tree.body
  )
  if not stmts or not isinstance(
      stmts[0], (ast.FunctionDef, ast.AsyncFunctionDef)
  ):
    raise WorkflowError("expected a function definition for workflow")

  fn_node = stmts[0]

  body_nodes = list(fn_node.body)
  # Strip leading docstring expression if present.
  if (
      body_nodes
      and isinstance(body_nodes[0], ast.Expr)
      and isinstance(body_nodes[0].value, ast.Constant)
      and isinstance(body_nodes[0].value.value, str)
  ):
    body_nodes = body_nodes[1:]

  if not body_nodes:
    raise WorkflowError("workflow function body cannot be empty")

  body_module = ast.Module(body=body_nodes, type_ignores=[])
  stripper = _WorkflowQualifierStripper()
  transformed_module = stripper.visit(body_module)
  ast.fix_missing_locations(transformed_module)
  script_source = ast.unparse(transformed_module) + "\n"

  validate_workflow_source(script_source)
  return script_source


@dataclasses.dataclass(frozen=True)
class WorkflowDefinition:
  """A validated workflow script extracted from a `@workflows.define` function.

  Attributes:
    name: Function name of the workflow.
    description: Docstring of the workflow function, if any.
    source: Validated standalone Python source for the `run_workflow` tool.
    fn: Original Python function.
  """

  name: str
  description: str
  source: str
  fn: Callable[[], Any]

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self.fn(*args, **kwargs)


def define(fn: Callable[[], Any]) -> WorkflowDefinition:
  r"""Decorator that validates a Python function as a `run_workflow` script.

  Example:
    ```python
    from google.antigravity.beta import workflows

    @workflows.define
    async def audit_repo():
      \"\"\"Audit repository modules in parallel.\"\"\"
      workflows.phase("Discover")
      modules = await workflows.agent(
          "List top-level modules",
          schema={"type": "array", "items": {"type": "string"}},
      )
      workflows.phase("Audit")
      findings = await workflows.parallel(
          modules,
          lambda m: workflows.agent(
              f"Audit module {m}",
              schema={"type": "string"},
              role=f"Audit {m}",
          ),
      )
      for f in findings:
        workflows.log(f)
    ```

  Args:
    fn: Zero-argument Python function (`async def` or `def`) using
      `run_workflow` primitives (`phase`, `log`, `agent`, `parallel`,
      `pipeline`).

  Returns:
    A `WorkflowDefinition` containing the validated script source.
  """
  source = extract_workflow_source(fn)
  doc = (inspect.getdoc(fn) or "").strip()
  return WorkflowDefinition(
      name=getattr(fn, "__name__", "workflow"),
      description=doc,
      source=source,
      fn=fn,
  )


_beta_lib.beta(sys.modules[__name__])
