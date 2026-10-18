"""One ``httpx.AsyncClient`` per service, and the HTTP → envelope mapping.

Every tool speaks through :func:`call`; what comes back is either the
decoded JSON body or one of two exceptions the guard turns into an
envelope (:mod:`geecs_mcp.guard`):

- :class:`ServiceUnavailable` — the service did not answer, or answered
  5xx: ``service_unreachable`` with ``service`` and ``url``.  The
  scanner's own ``manager_unreachable`` (its 503) lands here too, with
  the scanner's message — the manager being down is, from here, the
  scanner being unable to answer.
- :class:`ServiceRefusal` — a 4xx: the service's own kind and message,
  verbatim.  The scanner answers ``{"error": {"kind", "message", …}}``
  (its kinds are this server's kinds); the portal and the logbook answer
  FastAPI's ``{"detail": …}``, mapped by status (400/413/415/422 →
  ``invalid_request``, 404 → ``not_found``, 409 → ``policy_refusal``).
  A dict-shaped ``detail`` (the logbook's 409 carries the current entry;
  the portal's 409 carries the running job) rides along as ``extra``.

The test seam is :func:`transport_for`: the contract tests patch it to
hand each service an ``httpx.ASGITransport`` over that service's own
FastAPI app, so the tools run against real routes without a socket.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import anyio
import httpx

from geecs_mcp import config

logger = logging.getLogger("geecs_mcp.clients")

#: Per-request budgets.  Connect is short (same box or one hop); read is
#: generous for the share-touching routes; a caller passes a longer read
#: budget for the one verb that blocks (the scanner's stop waits up to
#: 120 s for the pause to land).
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 20.0

#: Status → kind for the ``{"detail": …}`` services (portal, logbook).
#: The envelope's own keys: context a service sends under these names
#: must not be splatted over them (a body with ``ok: true`` would otherwise
#: report a refusal as success).
_RESERVED_KEYS = frozenset({"ok", "error_kind", "message"})

_KIND_BY_STATUS = {
    400: "invalid_request",
    404: "not_found",
    409: "policy_refusal",
    413: "invalid_request",
    415: "invalid_request",
    422: "invalid_request",
}


class ServiceUnavailable(Exception):
    """The service did not answer, or answered with its own failure (5xx)."""

    def __init__(self, service: str, url: str, message: str) -> None:
        super().__init__(f"{service} at {url}: {message}")
        self.service = service
        self.url = url
        self.message = message


class ServiceTimeout(ServiceUnavailable):
    """The request was sent and the whole answer did not come within its budget.

    Still an outage to every caller that does not ask (the guard maps it
    as ``service_unreachable``); a verb whose request may have taken
    effect — a stop the scanner is still landing — catches it and says so.
    """


class ServiceRefusal(Exception):
    """A 4xx: the service's verdict on the request, kind and words verbatim."""

    def __init__(self, kind: str, message: str, **extra: Any) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.extra = extra


_clients: dict[str, httpx.AsyncClient] = {}
_lock = threading.Lock()


def transport_for(service: str) -> Optional[httpx.AsyncBaseTransport]:
    """The transport a service's client uses — ``None`` = real sockets.

    Patched by the contract tests to an ``ASGITransport`` over the
    service's in-process app.  Called through the module attribute.
    """
    return None


def get_client(service: str) -> httpx.AsyncClient:
    """The cached client for *service*; ``ServiceUnavailable`` when unconfigured."""
    base_url = config.service_url(service)
    if base_url is None:
        raise ServiceUnavailable(
            service,
            "(unconfigured)",
            f"no address for the {service}: set [mcp] {config.URL_KEYS[service]} "
            f"in {config.CONFIG_PATH}",
        )
    with _lock:
        client = _clients.get(service)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                transport=transport_for(service),
            )
            _clients[service] = client
        return client


async def reset_clients() -> None:
    """Close and forget every client (tests; config re-read on next use)."""
    with _lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        await client.aclose()


def _detail_text(body: Any, response: httpx.Response) -> str:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, dict) and isinstance(detail.get("message"), str):
            return detail["message"]
        if isinstance(detail, list):  # FastAPI validation errors
            return (
                "; ".join(
                    f"{'.'.join(str(p) for p in item.get('loc', []))}: {item.get('msg', '')}"
                    for item in detail
                    if isinstance(item, dict)
                )
                or response.text
            )
        if detail is not None:
            return str(detail)
    return response.text or f"HTTP {response.status_code}"


def _refusal(service: str, response: httpx.Response) -> ServiceRefusal:
    try:
        body = response.json()
    except ValueError:
        body = None
    if (
        service == "scanner"
        and isinstance(body, dict)
        and isinstance(body.get("error"), dict)
    ):
        error = dict(body["error"])
        kind = str(
            error.pop("kind", "")
            or _KIND_BY_STATUS.get(response.status_code, "invalid_request")
        )
        message = str(error.pop("message", "") or response.text)
        return ServiceRefusal(kind, message, **_context(error))
    kind = _KIND_BY_STATUS.get(response.status_code, "invalid_request")
    extra: dict[str, Any] = {}
    if isinstance(body, dict):
        # Context rides either beside ``detail`` (the portal's 409 carries
        # the active ``job`` there) or inside a dict-shaped ``detail`` (the
        # logbook's 409 carries the ``current`` entry): keep both.
        extra = {k: v for k, v in body.items() if k != "detail"}
        if isinstance(body.get("detail"), dict):
            extra.update({k: v for k, v in body["detail"].items() if k != "message"})
    return ServiceRefusal(kind, _detail_text(body, response), **_context(extra))


def _wall_clock(read_timeout_s: Optional[float]) -> float:
    """A request's whole budget: its read budget plus the connect."""
    read = READ_TIMEOUT_S if read_timeout_s is None else read_timeout_s
    return read + CONNECT_TIMEOUT_S


def _context(extra: dict[str, Any]) -> dict[str, Any]:
    """A service's refusal context minus the envelope's reserved keys."""
    return {k: v for k, v in extra.items() if k not in _RESERVED_KEYS}


async def call(
    service: str,
    method: str,
    path: str,
    *,
    params: Optional[dict[str, Any]] = None,
    json: Any = None,
    read_timeout_s: Optional[float] = None,
) -> Any:
    """One request to *service*; the decoded JSON body, or an exception above.

    ``params`` with ``None`` values are dropped (the services read absent,
    not ``null``).  A 204 or an empty body returns ``None``.
    """
    client = get_client(service)
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    timeout = (
        httpx.Timeout(read_timeout_s, connect=CONNECT_TIMEOUT_S)
        if read_timeout_s is not None
        else httpx.USE_CLIENT_DEFAULT
    )
    # httpx's read timeout bounds the gap between chunks, not the answer: a
    # peer trickling bytes would never trip it.  The read budget is also
    # the whole request's wall clock.
    limit = _wall_clock(read_timeout_s)
    try:
        with anyio.fail_after(limit):
            response = await client.request(
                method, path, params=clean_params, json=json, timeout=timeout
            )
    except (TimeoutError, httpx.ReadTimeout) as exc:
        raise ServiceTimeout(
            service,
            str(client.base_url),
            f"no complete answer within {limit:.0f} s: {type(exc).__name__}",
        ) from exc
    except httpx.HTTPError as exc:
        raise ServiceUnavailable(
            service, str(client.base_url), f"{type(exc).__name__}: {exc}"
        ) from exc
    if response.status_code >= 500:
        try:
            body = response.json()
        except ValueError:
            body = None
        raise ServiceUnavailable(
            service,
            str(client.base_url),
            f"HTTP {response.status_code}: {_detail_text(body, response)}",
        )
    if response.status_code >= 400:
        raise _refusal(service, response)
    if response.status_code == 204 or not response.content:
        return None
    try:
        return response.json()
    except ValueError as exc:
        # A 200 that is not JSON is the wrong thing answering on that
        # port (a proxy page, another service) — the deployment's fault,
        # never the request's.
        raise ServiceUnavailable(
            service,
            str(client.base_url),
            f"answered {response.status_code} with a non-JSON body "
            f"({response.headers.get('content-type', '?')})",
        ) from exc


async def get_bytes(
    service: str,
    path: str,
    *,
    params: Optional[dict[str, Any]] = None,
    max_bytes: int,
    read_timeout_s: Optional[float] = None,
) -> tuple[bytes, str, bool, Optional[int]]:
    """``(body, content type, truncated, total size)`` of one GET, capped at *max_bytes*.

    For the portal's artifact route: the body is streamed and reading
    stops once the cap is reached, so a large sidecar never lands whole in
    this process.  ``truncated`` is true only when bytes beyond the cap
    exist; ``total size`` is the ``Content-Length`` when the service sent
    one (the portal's file responses do), else the bytes read.
    """
    client = get_client(service)
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    timeout = (
        httpx.Timeout(read_timeout_s, connect=CONNECT_TIMEOUT_S)
        if read_timeout_s is not None
        else httpx.USE_CLIENT_DEFAULT
    )
    limit = _wall_clock(read_timeout_s)
    try:
        with anyio.fail_after(limit):
            return await _capped_read(
                client, service, path, clean_params, timeout, max_bytes
            )
    except (TimeoutError, httpx.ReadTimeout) as exc:
        raise ServiceTimeout(
            service,
            str(client.base_url),
            f"no complete answer within {limit:.0f} s: {type(exc).__name__}",
        ) from exc
    except httpx.HTTPError as exc:
        raise ServiceUnavailable(
            service, str(client.base_url), f"{type(exc).__name__}: {exc}"
        ) from exc


async def _capped_read(
    client: httpx.AsyncClient,
    service: str,
    path: str,
    clean_params: dict[str, Any],
    timeout: Any,
    max_bytes: int,
) -> tuple[bytes, str, bool, Optional[int]]:
    """The streamed GET behind :func:`get_bytes`, inside its wall clock."""
    async with client.stream(
        "GET", path, params=clean_params, timeout=timeout
    ) as response:
        if response.status_code >= 500:
            await response.aread()
            raise ServiceUnavailable(
                service, str(client.base_url), f"HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            await response.aread()
            raise _refusal(service, response)
        chunks: list[bytes] = []
        total = 0
        truncated = False
        async for chunk in response.aiter_bytes():
            room = max_bytes - total
            if len(chunk) > room:
                chunks.append(chunk[:room])
                total += room
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
        content_type = response.headers.get("content-type", "application/octet-stream")
        declared = response.headers.get("content-length")
    size: Optional[int] = total
    if declared is not None and declared.isdigit():
        size = int(declared)
        truncated = truncated or size > total
    return b"".join(chunks), content_type, truncated, size
