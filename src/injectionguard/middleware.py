"""FastAPI middleware for prompt injection detection.

Usage::

    from fastapi import FastAPI
    from injectionguard.middleware import InjectionGuardMiddleware

    app = FastAPI()
    app.add_middleware(InjectionGuardMiddleware, fail_on="high")
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from injectionguard.detector import Detector
from injectionguard.types import ThreatLevel, LEVEL_ORDER


class InjectionGuardMiddleware:
    """ASGI middleware that scans request bodies for prompt injection.

    Designed for FastAPI/Starlette but works with any ASGI app.
    Scans JSON request bodies and blocks requests whose threat level
    meets or exceeds ``fail_on``.

    Parameters
    ----------
    app:
        The ASGI application.
    fail_on:
        Minimum threat level that causes a 400 rejection.
        One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        Default ``"high"``.
    scan_paths:
        Optional list of URL path prefixes to scan. If ``None``, all
        POST/PUT/PATCH requests are scanned.
    allow_list:
        Strings that the detector should ignore (passed through to
        detector configuration, future use).
    on_detection:
        Optional callback ``(request_path, result) -> None`` invoked when
        an injection is detected (for logging).
    """

    def __init__(
        self,
        app: Any,
        fail_on: str = "high",
        scan_paths: Optional[list[str]] = None,
        allow_list: Optional[list[str]] = None,
        on_detection: Optional[Callable] = None,
    ):
        self.app = app
        self.fail_on = ThreatLevel(fail_on)
        self.scan_paths = scan_paths
        self.allow_list = allow_list or []
        self.on_detection = on_detection
        self.detector = Detector(threshold=ThreatLevel.LOW)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        if method not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if self.scan_paths and not any(path.startswith(p) for p in self.scan_paths):
            await self.app(scope, receive, send)
            return

        # Buffer the request body
        body_parts: list[bytes] = []
        receive_done = False

        async def buffered_receive() -> dict:
            nonlocal receive_done
            if body_parts and receive_done:
                # Replay already-read body
                return {"type": "http.request", "body": b"".join(body_parts), "more_body": False}
            msg = await receive()
            if msg["type"] == "http.request":
                body_parts.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    receive_done = True
            return msg

        # Read the full body
        while not receive_done:
            await buffered_receive()

        full_body = b"".join(body_parts)

        # Extract text fields from JSON body
        texts = _extract_texts(full_body)

        if texts:
            combined = " ".join(texts)
            result = self.detector.scan(combined)

            if not result.is_safe:
                result_level_idx = LEVEL_ORDER.index(result.threat_level)
                fail_idx = LEVEL_ORDER.index(self.fail_on)

                if result_level_idx >= fail_idx:
                    if self.on_detection:
                        try:
                            self.on_detection(path, result)
                        except Exception:
                            pass

                    resp_body = json.dumps({
                        "error": "Request blocked: potential prompt injection detected",
                        "threat_level": result.threat_level.value,
                        "detections": len(result.detections),
                    }).encode()

                    await send({
                        "type": "http.response.start",
                        "status": 400,
                        "headers": [
                            [b"content-type", b"application/json"],
                            [b"content-length", str(len(resp_body)).encode()],
                        ],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": resp_body,
                    })
                    return

        # Replay body for the actual app
        body_sent = False

        async def replay_receive() -> dict:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": full_body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


def _extract_texts(body: bytes) -> list[str]:
    """Extract string values from a JSON body."""
    if not body:
        return []
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    return _collect_strings(data)


def _collect_strings(obj: Any, depth: int = 0) -> list[str]:
    """Recursively collect string values from a JSON structure."""
    if depth > 10:
        return []
    texts: list[str] = []
    if isinstance(obj, str):
        texts.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            texts.extend(_collect_strings(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(_collect_strings(item, depth + 1))
    return texts
