# Multi-Agent Workflows Example (`@beta.workflows.define` & `agent.beta.run_workflow`)

This example demonstrates how to author and execute deterministic multi-agent
workflows using `from google.antigravity import beta` (`beta.workflows`) and
`await agent.beta.run_workflow(...)`.

## 1. Authoring Typed Workflows with `@beta.workflows.define`

Decorate a zero-argument Python function (`async def` or `def`) with
`@beta.workflows.define` (or `@workflows.define` via
`workflows = beta.workflows`) and call the five workflow primitives
(`workflows.phase`, `workflows.log`, `workflows.agent`, `workflows.parallel`,
`workflows.pipeline`). The SDK validates the function's AST at definition time
and rewrites `workflows.<primitive>` calls for the sandboxed `run_workflow`
Python engine:

```python
from google.antigravity import Agent, LocalAgentConfig, beta, types

workflows = beta.workflows


@workflows.define
async def security_audit_workflow() -> None:
  """Audits source snippets in parallel and synthesizes an executive summary."""
  workflows.phase("Parallel File Audit")
  snippets = [
      {
          "file": "auth.py",
          "code": "jwt.decode(token, options={'verify_signature': False})",
      },
      {
          "file": "db.py",
          "code": "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')",
      },
  ]
  findings = await workflows.parallel(
      snippets,
      lambda s: workflows.agent(
          f"Audit `{s['file']}` (`{s['code']}`) in one concise sentence"
          " naming the vulnerability.",
          schema={"type": "string"},
          role=f"Audit {s['file']}",
          type_name="file_auditor",
      ),
  )

  workflows.phase("Executive Synthesis")
  combined = "\n".join(str(f).strip() for f in findings)
  summary = await workflows.agent(
      f"Summarize these security findings:\n{combined}",
      schema={
          "type": "object",
          "properties": {
              "total_issues": {"type": "integer"},
              "highest_severity": {"type": "string"},
              "summary": {"type": "string"},
          },
          "required": ["total_issues", "highest_severity", "summary"],
      },
      role="Lead Synthesizer",
      type_name="lead_synthesizer",
  )
  workflows.log(
      f"Audit complete: {summary['total_issues']} issues"
      f" (highest={summary['highest_severity']}): {summary['summary']}"
  )


file_auditor = types.SubagentConfig(
    name="file_auditor",
    description="Audits a code snippet for security vulnerabilities.",
    system_instructions="Identify the vulnerability in one sentence.",
    capabilities=types.SubagentCapabilities(enabled_tools=[]),
)
lead_synthesizer = types.SubagentConfig(
    name="lead_synthesizer",
    description="Synthesizes security findings into a structured report.",
    system_instructions="Combine findings and respond with the JSON schema.",
    capabilities=types.SubagentCapabilities(enabled_tools=[]),
)

config = LocalAgentConfig(subagents=[file_auditor, lead_synthesizer])

async with Agent(config) as my_agent:
  result = await my_agent.beta.run_workflow(security_audit_workflow)
  print("Output:\n", result.output)
  print("Response:", result.response_text)
```

## 2. Running a Standalone Workflow Script File

You can also pass a `.py` file path (`str` or `pathlib.Path`) via
`await agent.beta.run_workflow(script_path=..., description=...)`:

```python
import pathlib

script_path = pathlib.Path("release_checklist_workflow.py")
script_path.write_text("""\
phase("Sequential Verification")
checks = ["unit_tests", "lint_check"]
results = await pipeline(
    checks,
    lambda item: agent(
        f"Confirm check '{item}' passed.",
        schema={"type": "string"},
        role=f"Verify {item}",
    ),
)
for item_name, res in zip(checks, results):
    log(f"{item_name}: {str(res).strip()}")
""")

async with Agent(LocalAgentConfig()) as my_agent:
  result = await my_agent.beta.run_workflow(
      script_path=script_path,
      description="Run the sequential release verification checklist",
  )
  print(result.output)
```

## Workflow Sandbox Rules

Workflow scripts execute in a restricted Python environment:

- **Top-level `await` and `async def` helpers**: `await` works at the top level
  of the script, and nested `async def` stage helpers can be passed to
  `parallel` or `pipeline`.
- **No `import` statements** and **no `while` loops** (use `for` loops over
  bounded collections).
- **No leading-underscore attribute access** (`obj._attr`) or **`__` dunder
  names**.
- **Builtins**: Only pure builtins (`abs`, `all`, `any`, `bool`, `dict`,
  `enumerate`, `float`, `int`, `len`, `list`, `max`, `min`, `range`, `reversed`,
  `sorted`, `str`, `tuple`, `zip`) plus the 5 workflow primitives (`phase`,
  `log`, `agent`, `parallel`, `pipeline`) are available. Use `log(...)` instead
  of `print(...)`.
