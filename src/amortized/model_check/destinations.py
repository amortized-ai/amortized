"""Destination policy for requests the platform sends to user-named model endpoints.

Applied to the final URL (after provider resolution) and every address it resolves to:

- loopback, link-local, unspecified, multicast, reserved and cloud-metadata addresses: refused;
- public HTTPS (every resolved address public): allowed;
- anything else (private addresses, plain HTTP): only under an operator-approved prefix.

Connections then go only to the vetted addresses (``pinned_transport``), so DNS can't be rebound
in between; TLS still verifies the original hostname. A host that doesn't resolve is
*unresolved*, not denied, since DNS can fail transiently.

Depends only on the stdlib, httpx and httpcore: it ships in the standalone checker image.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

import httpcore
import httpx

ALLOWED = "allowed"
DENIED = "denied"
UNRESOLVED = "unresolved"

# Cloud instance-metadata endpoints outside the link-local ranges (169.254.169.254 itself is
# link-local): AWS IPv6, Alibaba Cloud.
_METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(a) for a in ("fd00:ec2::254", "100.100.100.200")
)
_DEFAULT_PORTS = {"http": 80, "https": 443}


AddressClass = Literal["blocked", "private", "public"]


def classify(address: str) -> AddressClass:
    """``blocked`` (never allowed), ``private`` (needs approval) or ``public``."""
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped  # judge ::ffff:10.0.0.1 as 10.0.0.1
    if (
        ip in _METADATA_ADDRESSES
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
    ):
        return "blocked"
    return "public" if ip.is_global else "private"


@dataclass(frozen=True)
class Decision:
    verdict: str  # ALLOWED | DENIED | UNRESOLVED
    reason: str = ""  # why a destination was denied or unresolved (no secrets, no full URL)
    addresses: tuple[str, ...] = field(default=())  # the vetted addresses, when allowed

    @property
    def allowed(self) -> bool:
        return self.verdict == ALLOWED


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 dot-segment removal — what httpx does to a path before sending it."""
    out: list[str] = []
    for segment in path.split("/"):
        if segment == "..":
            if len(out) > 1:
                out.pop()
        elif segment != ".":
            out.append(segment)
    return "/".join(out)


def _canonical(url: str) -> tuple[str, str, int, str] | None:
    """(scheme, host, port, path) as the request will actually go out, or None if the URL is
    unusable or its path is ambiguous (percent-encoded dots or slashes)."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    if any(token in parts.path.lower() for token in ("%2e", "%2f", "%5c", "\\")):
        return None
    path = _remove_dot_segments(parts.path or "/").rstrip("/")
    return scheme, parts.hostname.lower().rstrip("."), port or _DEFAULT_PORTS[scheme], path


def within_prefix(url: str, prefix: str) -> bool:
    """Whether ``url`` falls under ``prefix``: same scheme, host and port, and a path at or
    below the prefix's at a segment boundary — compared after normalization, as sent."""
    u, p = _canonical(url), _canonical(prefix)
    if u is None or p is None or u[:3] != p[:3]:
        return False
    return not p[3] or u[3] == p[3] or u[3].startswith(p[3] + "/")


def is_approved(url: str, approved: Sequence[str]) -> bool:
    return "*" in approved or any(within_prefix(url, prefix) for prefix in approved if prefix)


def decide(url: str, addresses: Sequence[str], approved: Sequence[str]) -> Decision:
    """The policy decision for ``url`` given the addresses its host resolved to."""
    canonical = _canonical(url)
    if canonical is None:
        return Decision(DENIED, "isn't a valid http(s) URL")
    if not addresses:
        return Decision(UNRESOLVED, "its host name didn't resolve")
    classes = [classify(a) for a in addresses]
    if "blocked" in classes:
        return Decision(DENIED, "resolves to a loopback, link-local or metadata address")
    if all(c == "public" for c in classes) and canonical[0] == "https":
        return Decision(ALLOWED, addresses=tuple(addresses))
    if is_approved(url, approved):
        return Decision(ALLOWED, addresses=tuple(addresses))
    if canonical[0] == "http" and all(c == "public" for c in classes):
        return Decision(DENIED, "isn't HTTPS (a public endpoint must use HTTPS)")
    return Decision(DENIED, "is a private address that an operator hasn't approved")


def literal_address(url: str) -> str | None:
    """The address when the URL's host is an IP literal (nothing to resolve), else None."""
    canonical = _canonical(url)
    if canonical is None:
        return None
    try:
        return str(ipaddress.ip_address(canonical[1].strip("[]")))
    except ValueError:
        return None


async def resolve(url: str) -> list[str]:
    """Every address the URL's host resolves to (IPv4 and IPv6); [] when it doesn't resolve."""
    canonical = _canonical(url)
    if canonical is None:
        return []
    literal = literal_address(url)
    if literal is not None:
        return [literal]
    _, host, port, _ = canonical
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )
    except (OSError, UnicodeError):
        return []
    seen: dict[str, None] = {}
    for info in infos:
        seen[str(info[4][0]).split("%")[0]] = None  # drop an IPv6 zone id
    return list(seen)


async def authorize(url: str, approved: Sequence[str]) -> Decision:
    """Resolve ``url`` and decide whether a request may be sent to it."""
    return decide(url, await resolve(url), approved)


def chat_request_url(base_url: str, provider_type: str) -> str:
    """The URL a chat request to an endpoint goes to — OpenAI ``/chat/completions`` or the
    Anthropic Messages API — built one way for the checker, SDG dispatch and (mirrored) the eval
    runner, so the policy authorizes the route actually requested everywhere."""
    base = base_url.rstrip("/")
    if provider_type == "anthropic":
        return base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    return base + "/chat/completions"


IDENTITY_ENCODING = {"Accept-Encoding": "identity"}


async def read_limited(resp: httpx.Response, limit: int) -> tuple[bytes, bool]:
    """Up to ``limit`` bytes of a streamed reply's decoded body, and whether there was more.

    Bounded before decoding too: raw bytes are read with the same cap (a declared length over it
    stops at once), and gzip/deflate are decompressed with an output limit, so a small
    compressed reply can't expand in memory. Other encodings are refused (returned as truncated).
    """
    declared = resp.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return b"", True
    raw = bytearray()
    try:
        async for chunk in resp.aiter_raw():
            raw.extend(chunk)
            if len(raw) > limit:
                return bytes(raw[:limit]), True
    except httpx.StreamConsumed:
        # Already buffered and decoded (an in-memory transport).
        body = resp.content
        return body[:limit], len(body) > limit
    encoding = resp.headers.get("content-encoding", "identity").strip().lower()
    if encoding in ("", "identity"):
        return bytes(raw), False
    if encoding not in ("gzip", "x-gzip", "deflate"):
        return b"", True
    # 32 + MAX_WBITS auto-detects gzip and zlib headers; raw deflate needs negative wbits.
    if encoding == "deflate":
        candidates: tuple[int, ...] = (zlib.MAX_WBITS, -zlib.MAX_WBITS)
    else:
        candidates = (32 + zlib.MAX_WBITS,)
    for wbits in candidates:
        decoder = zlib.decompressobj(wbits)
        try:
            body = decoder.decompress(bytes(raw), limit + 1)
        except zlib.error:
            continue
        if len(body) > limit or decoder.unconsumed_tail:
            return body[:limit], True
        return body, False
    return b"", True


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """Opens TCP connections only to the vetted addresses, whatever host name httpcore asks
    for; TLS is still negotiated (and verified) for the request's original host name."""

    def __init__(self, addresses: Sequence[str]) -> None:
        self._addresses = tuple(addresses)
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> httpcore.AsyncNetworkStream:
        last: Exception | None = None
        for address in self._addresses:
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,  # type: ignore[arg-type]
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last = exc
        raise last or httpcore.ConnectError("no vetted address to connect to")

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: object = None
    ) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError("unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class _PinnedTransport(httpx.AsyncHTTPTransport):
    def __init__(self, addresses: Sequence[str], verify: str | bool = True) -> None:
        super().__init__(verify=verify)
        ssl_context = self._pool._ssl_context
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context, network_backend=_PinnedBackend(addresses)
        )


def pinned_transport(decision: Decision, verify: str | bool = True) -> httpx.AsyncBaseTransport:
    """An httpx transport that connects only to an allowed decision's vetted addresses."""
    if not decision.allowed:
        raise ValueError("pinned_transport needs an allowed decision")
    return _PinnedTransport(decision.addresses, verify=verify)
