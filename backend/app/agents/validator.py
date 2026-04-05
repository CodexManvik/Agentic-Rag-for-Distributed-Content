import re
from typing import Any


class AgentValidationError(ValueError):
    pass


def _is_type_match(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate_manifest_structure(manifest: dict[str, Any]) -> None:
    required_top_level = ["name", "version", "description", "inputs", "outputs", "execution"]
    missing = [k for k in required_top_level if k not in manifest]
    if missing:
        raise AgentValidationError(f"Missing required fields: {missing}")

    if not isinstance(manifest["name"], str) or not manifest["name"].strip():
        raise AgentValidationError("Agent name must be a non-empty string")

    if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", str(manifest["version"])):
        raise AgentValidationError("Version must use semantic format x.y.z")

    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise AgentValidationError("execution must be an object")

    execution_type = execution.get("type")
    if execution_type not in {"python", "workflow", "tool_call"}:
        raise AgentValidationError("execution.type must be one of: python, workflow, tool_call")

    if execution_type == "python":
        entrypoint = execution.get("entrypoint")
        if not isinstance(entrypoint, str) or ":" not in entrypoint:
            raise AgentValidationError("execution.entrypoint must be in 'module.path:function_name' format")

    if execution_type == "workflow":
        mode = execution.get("execution_mode")
        if mode not in {"sequence", "parallel"}:
            raise AgentValidationError("workflow execution_mode must be 'sequence' or 'parallel'")
        workflow = execution.get("workflow")
        if not isinstance(workflow, list) or len(workflow) == 0:
            raise AgentValidationError("workflow execution requires non-empty 'workflow' list")
        for step in workflow:
            if not isinstance(step, dict) or not isinstance(step.get("agent"), str):
                raise AgentValidationError("each workflow step must include string field 'agent'")

    if execution_type == "tool_call":
        tool_name = execution.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise AgentValidationError("tool_call execution requires non-empty 'tool_name'")


def validate_inputs(manifest: dict[str, Any], inputs: dict[str, Any]) -> None:
    input_spec = manifest.get("inputs", {})
    if not isinstance(input_spec, dict):
        raise AgentValidationError("inputs must be an object")

    for field_name, field_spec in input_spec.items():
        if not isinstance(field_spec, dict):
            continue

        is_required = bool(field_spec.get("required", False))
        expected_type = field_spec.get("type")

        if is_required and field_name not in inputs:
            raise AgentValidationError(f"Missing required input: {field_name}")

        if field_name in inputs and isinstance(expected_type, str):
            if not _is_type_match(inputs[field_name], expected_type):
                raise AgentValidationError(
                    f"Invalid type for input '{field_name}': expected {expected_type}"
                )
