from importlib import import_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from app.agents.context import ExecutionContext
from app.agents.orchestrator import AgentOrchestrator
from app.agents.registry import AgentManifest, AgentRegistry
from app.agents.validator import AgentValidationError, validate_inputs
from app.graph.state import NavigatorState
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.whitelist import ToolWhitelist


class AgentExecutor:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        tool_manifest_dir = Path(__file__).resolve().parents[1] / "tools" / "manifests"
        self.tool_registry = ToolRegistry(tool_manifest_dir)
        self.tool_registry.load_tools()
        self.tool_whitelist = ToolWhitelist(initially_enabled=[])
        self.tool_executor = ToolExecutor(self.tool_registry, self.tool_whitelist)
        self.orchestrator = AgentOrchestrator(self)

    def append_trace(
        self,
        state: Any,
        *,
        node: str,
        status: str,
        detail: str,
        context: ExecutionContext,
        tool_name: str | None = None,
    ) -> None:
        trace_item: dict[str, Any] = {
            "node": node,
            "status": status,
            "detail": detail,
            "ts": datetime.now(timezone.utc).isoformat(),
            "execution_id": context.execution_id,
            "parent_execution_id": context.parent_execution_id,
            "root_execution_id": context.root_execution_id,
            "execution_depth": context.depth,
            "execution_path": list(context.path),
        }
        if tool_name:
            trace_item["tool"] = tool_name

        trace_list = state.get("trace")
        if not isinstance(trace_list, list):
            state["trace"] = []
            trace_list = state["trace"]
        trace_list.append(trace_item)

    def _load_entrypoint(self, entrypoint: str) -> Callable[[NavigatorState], NavigatorState]:
        module_path, function_name = entrypoint.split(":", 1)
        module = import_module(module_path)
        fn = getattr(module, function_name, None)
        if fn is None or not callable(fn):
            raise AgentValidationError(f"Entrypoint not callable: {entrypoint}")
        return cast(Callable[[NavigatorState], NavigatorState], fn)

    def execute(self, name: str, inputs: dict[str, Any], version: str | None = None) -> NavigatorState:
        root_context = ExecutionContext.root(name)
        return self.execute_with_context(name, inputs, root_context, version)

    def execute_with_context(
        self,
        name: str,
        inputs: Any,
        context: ExecutionContext,
        version: str | None = None,
    ) -> NavigatorState:
        manifest = self.registry.get_agent(name, version)
        validate_inputs(manifest.raw, inputs)
        self.append_trace(
            inputs,
            node=f"agent:{name}",
            status="start",
            detail=f"starting execution type={manifest.execution.get('type')}",
            context=context,
        )

        execution_type = manifest.execution.get("type")
        if execution_type == "python":
            out = self._execute_python(manifest, inputs)
        elif execution_type == "workflow":
            out = self._execute_workflow(manifest, inputs, context)
        elif execution_type == "tool_call":
            out = self._execute_tool_call(manifest, inputs, context)
        else:
            raise AgentValidationError(f"Unsupported execution type: {execution_type}")

        self.append_trace(
            out,
            node=f"agent:{name}",
            status="ok",
            detail="execution completed",
            context=context,
        )
        return cast(NavigatorState, out)

    def _execute_python(self, manifest: AgentManifest, inputs: Any) -> NavigatorState:
        entrypoint = manifest.execution.get("entrypoint")
        if not isinstance(entrypoint, str):
            raise AgentValidationError("Missing execution.entrypoint")

        fn = self._load_entrypoint(entrypoint)
        return fn(inputs)  # type: ignore[arg-type, return-value]

    def _execute_workflow(
        self,
        manifest: AgentManifest,
        inputs: Any,
        context: ExecutionContext,
    ) -> NavigatorState:
        return cast(NavigatorState, self.orchestrator.execute_workflow(manifest, inputs, context))

    def _execute_tool_call(
        self,
        manifest: AgentManifest,
        inputs: Any,
        context: ExecutionContext,
    ) -> NavigatorState:
        execution = manifest.execution
        tool_name = execution.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise AgentValidationError("tool_call execution missing tool_name")

        input_field = str(execution.get("input_field", "query"))
        output_field = str(execution.get("output_field", "tool_result"))
        tool_args = {"text": str(inputs.get(input_field, ""))}

        self.append_trace(
            inputs,
            node=f"tool:{tool_name}",
            status="start",
            detail="tool invocation requested",
            context=context,
            tool_name=tool_name,
        )

        result = self.tool_executor.execute(tool_name, tool_args)
        output = dict(inputs)
        output[output_field] = result

        self.append_trace(
            output,
            node=f"tool:{tool_name}",
            status="ok",
            detail="tool invocation completed",
            context=context,
            tool_name=tool_name,
        )
        return output  # type: ignore[return-value]

    def create_node(self, manifest: AgentManifest) -> Callable[[NavigatorState], NavigatorState]:
        def _node(state: NavigatorState) -> NavigatorState:
            validate_inputs(manifest.raw, dict(state))
            context = ExecutionContext.root(manifest.name)
            return self.execute_with_context(manifest.name, dict(state), context, manifest.version)

        return _node
