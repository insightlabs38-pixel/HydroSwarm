"""Shared bounded-inference and permission machinery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from .schemas import FallbackMode, ToolPermission


OutputT = TypeVar("OutputT", bound=BaseModel)


class AgentPermissionError(PermissionError):
    pass


class AgentTimeoutError(TimeoutError):
    pass


class DeterministicAgent(Generic[OutputT]):
    output_type: type[OutputT]
    permissions: frozenset[ToolPermission] = frozenset()
    visible_fields: frozenset[str] = frozenset()

    def __init__(
        self,
        inference: Callable[[Mapping[str, object]], OutputT | Mapping[str, object]] | None = None,
        *,
        fallback_inference: Callable[[Mapping[str, object]], OutputT | Mapping[str, object]] | None = None,
    ) -> None:
        self.inference = inference
        self.fallback_inference = fallback_inference
        self.last_mode = FallbackMode.FULL_HYBRID
        self.last_error: str | None = None

    def require(self, permission: ToolPermission) -> None:
        if permission not in self.permissions:
            raise AgentPermissionError(f"{type(self).__name__} cannot use {permission}")

    def visible_state(self, state: Mapping[str, object]) -> dict[str, object]:
        return {key: state[key] for key in sorted(self.visible_fields) if key in state}

    @staticmethod
    def _bounded_call(
        function: Callable[[Mapping[str, object]], Any],
        visible: Mapping[str, object],
        timeout_seconds: float,
    ) -> Any:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hydroswarm-agent")
        future = executor.submit(function, visible)
        try:
            value = future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise AgentTimeoutError("agent inference exceeded its deadline") from exc
        except BaseException:
            executor.shutdown(wait=True)
            raise
        else:
            executor.shutdown(wait=True)
            return value

    def invoke(self, state: Mapping[str, object], *, timeout_seconds: float) -> OutputT:
        self.require(ToolPermission.RUN_INFERENCE)
        visible = self.visible_state(state)
        self.last_mode = FallbackMode.FULL_HYBRID
        self.last_error = None
        if self.inference is not None:
            try:
                raw = self._bounded_call(self.inference, visible, timeout_seconds)
                return self.output_type.model_validate(raw)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
        if self.fallback_inference is not None:
            try:
                raw = self._bounded_call(self.fallback_inference, visible, timeout_seconds)
                output = self.output_type.model_validate(raw)
                self.last_mode = FallbackMode.DEGRADED_HYBRID
                return output
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
        self.last_mode = FallbackMode.CLASSICAL_SAFE
        return self.deterministic_fallback(visible)

    def deterministic_fallback(self, state: Mapping[str, object]) -> OutputT:
        raise NotImplementedError
