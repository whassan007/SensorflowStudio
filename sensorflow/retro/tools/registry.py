"""Tool registry: the MCP-style boundary between the agent and the world.

Every tool declares an input schema, an output schema, permissions
(read_only, default True), a timeout, and error behavior. Every call —
success or failure — is appended to an audit trail with the arguments, a
SHA-256 hash of the result, and a UTC timestamp. Write tools additionally
require an explicit policy-authorization flag per call.

If the `mcp` package is installed, tools/mcp_server.py exposes this same
registry as a real MCP server; otherwise this registry itself is the
MCP-style boundary.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from sensorflow.retro import store


class ToolPermissionError(PermissionError):
    pass


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    read_only: bool = True
    timeout_s: float = 10.0
    error_behavior: str = "raise"  # raise | return_error


class AuditRecord(BaseModel):
    call_id: str
    analysis_id: str
    tool: str
    timestamp: str
    args: Dict[str, Any]
    status: str                    # ok | error | timeout | denied
    result_hash: Optional[str]     # sha256 of canonical result JSON
    elapsed_ms: float
    error: Optional[str] = None
    authorized_write: bool = False


class _RegisteredTool(BaseModel):
    spec: ToolSpec
    fn: Callable[..., BaseModel]
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]

    model_config = {"arbitrary_types_allowed": True}


def _hash_result(result: Any) -> str:
    blob = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class ToolCallResult(BaseModel):
    ok: bool
    tool: str
    call_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


class ToolRegistry:
    """Audited, permissioned, schema-validated tool dispatch."""

    def __init__(self, analysis_id: Optional[str] = None,
                 persist_audit: bool = True):
        self._tools: Dict[str, _RegisteredTool] = {}
        self.analysis_id = analysis_id or f"adhoc-{uuid.uuid4().hex[:8]}"
        self.persist_audit = persist_audit
        self.audit_log: List[AuditRecord] = []

    # ------------------------------------------------------------ registration

    def register(self, name: str, fn: Callable[..., BaseModel],
                 input_model: Type[BaseModel], output_model: Type[BaseModel],
                 description: str, read_only: bool = True,
                 timeout_s: float = 10.0, error_behavior: str = "raise") -> None:
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        if error_behavior not in ("raise", "return_error"):
            raise ValueError("error_behavior must be 'raise' or 'return_error'")
        spec = ToolSpec(name=name, description=description,
                        input_schema=input_model.model_json_schema(),
                        output_schema=output_model.model_json_schema(),
                        read_only=read_only, timeout_s=timeout_s,
                        error_behavior=error_behavior)
        self._tools[name] = _RegisteredTool(spec=spec, fn=fn,
                                            input_model=input_model,
                                            output_model=output_model)

    def specs(self) -> List[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def get_spec(self, name: str) -> ToolSpec:
        return self._tools[name].spec

    # ------------------------------------------------------------------ audit

    def _audit(self, record: AuditRecord) -> None:
        self.audit_log.append(record)
        if self.persist_audit:
            store.append_audit(self.analysis_id, record.model_dump())

    # ------------------------------------------------------------------- call

    def call(self, name: str, args: Dict[str, Any],
             policy_authorization: bool = False) -> ToolCallResult:
        call_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        ts = datetime.now(timezone.utc).isoformat()

        def audit(status: str, result_hash: Optional[str] = None,
                  error: Optional[str] = None) -> None:
            self._audit(AuditRecord(
                call_id=call_id, analysis_id=self.analysis_id, tool=name,
                timestamp=ts, args=args, status=status, result_hash=result_hash,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
                error=error, authorized_write=policy_authorization))

        if name not in self._tools:
            audit("error", error="unknown tool")
            raise KeyError(f"unknown tool {name!r}")
        tool = self._tools[name]

        if not tool.spec.read_only and not policy_authorization:
            audit("denied", error="write tool requires policy_authorization=True")
            raise ToolPermissionError(
                f"tool {name!r} is a WRITE tool and requires an explicit "
                "policy_authorization=True flag on the call")

        try:
            validated = tool.input_model(**args)
        except ValidationError as exc:
            audit("error", error=f"input validation failed: {exc}")
            if tool.spec.error_behavior == "return_error":
                return ToolCallResult(ok=False, tool=name, call_id=call_id,
                                      error=f"input validation failed: {exc}",
                                      elapsed_ms=(time.perf_counter() - t0) * 1000)
            raise

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(tool.fn, validated)
                output = future.result(timeout=tool.spec.timeout_s)
        except concurrent.futures.TimeoutError:
            audit("timeout", error=f"timed out after {tool.spec.timeout_s}s")
            if tool.spec.error_behavior == "return_error":
                return ToolCallResult(ok=False, tool=name, call_id=call_id,
                                      error=f"timeout after {tool.spec.timeout_s}s",
                                      elapsed_ms=(time.perf_counter() - t0) * 1000)
            raise TimeoutError(f"tool {name!r} timed out after {tool.spec.timeout_s}s")
        except Exception as exc:
            audit("error", error=f"{exc.__class__.__name__}: {exc}")
            if tool.spec.error_behavior == "return_error":
                return ToolCallResult(ok=False, tool=name, call_id=call_id,
                                      error=f"{exc.__class__.__name__}: {exc}",
                                      elapsed_ms=(time.perf_counter() - t0) * 1000)
            raise

        if not isinstance(output, tool.output_model):
            output = tool.output_model(**(output if isinstance(output, dict)
                                          else output.model_dump()))
        result_dict = output.model_dump()
        audit("ok", result_hash=_hash_result(result_dict))
        return ToolCallResult(ok=True, tool=name, call_id=call_id,
                              result=result_dict,
                              elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))
