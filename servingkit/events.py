"""A structured event bus with pluggable log hooks.

Everything in this package that takes time or makes a decision emits an event here, and anything
that wants to watch subscribes a hook. That is the whole interface: `emit()` on one side,
`subscribe()` on the other, and no component knows who is listening.

    from servingkit.events import BUS, JsonlHook
    BUS.subscribe(JsonlHook())                       # one JSON object per line on stdout
    BUS.subscribe(FileHook("/var/log/servingkit.jsonl"))
    BUS.subscribe(WebhookHook("https://collector/ingest"))

Or from the environment, which is how the container does it:

    SERVINGKIT_LOG_HOOKS="jsonl:-,file:/var/log/sk.jsonl,webhook:https://collector/ingest,ring:512"

Two properties are the reason this is a module rather than a `print`:

**A hook that fails must not fail the run.** Logging is not the work. Every hook call is wrapped,
the exception is counted against that hook, and a hook that keeps raising is *muted* after a
threshold rather than being allowed to raise on every event forever. `BUS.health()` reports the
error and mute counts, so a silently broken collector is visible instead of invisible.

**A hook that blocks is worse than a hook that raises.** A webhook to a collector that has
stopped answering would otherwise turn an observability feature into a latency outage: every
`emit()` in the request path would wait on someone else's TCP timeout. So hooks that talk to the
network are wrapped in `Async`, which hands the event to a worker thread through a *bounded*
queue and **drops** — counting the drops — when the queue is full. Dropping telemetry is the
correct failure mode; blocking the thing being measured is not.

`seq` is monotonic per bus, so a consumer can detect a gap rather than having to trust that it
received everything. That is what makes dropping acceptable.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# Levels, ordered. A hook declares the least severe level it wants.
LEVELS = ("debug", "info", "warn", "error")


@dataclass(frozen=True)
class Event:
    """One thing that happened. Flat, JSON-encodable, and self-describing.

    `kind` is dotted and hierarchical — `pipeline.stage.ok`, `agent.decide`, `http.request` — so a
    hook can filter on a prefix without a schema registry.
    """

    seq: int
    ts: float
    kind: str
    level: str = "info"
    run: str | None = None          # groups the events of one pipeline or agent run
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.ts))
        return d

    def as_json(self) -> str:
        # `default=str` rather than a failure: an event carrying something unserializable is a
        # logging bug, and losing the event to a TypeError is a worse outcome than a repr.
        return json.dumps(self.as_dict(), default=str, allow_nan=False)

    def line(self) -> str:
        """A human-readable single line, for a terminal."""
        head = f"{time.strftime('%H:%M:%S', time.localtime(self.ts))} {self.kind:<22}"
        body = " ".join(f"{k}={_short(v)}" for k, v in self.data.items())
        return f"{head} {body}".rstrip()


def _short(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


# --------------------------------------------------------------------------------- hooks
class Hook:
    """A log sink. Subclasses implement `write`; the bus handles filtering and failure."""

    name = "hook"
    min_level = "debug"
    prefixes: tuple[str, ...] = ()      # empty means every kind

    def wants(self, ev: Event) -> bool:
        if LEVELS.index(ev.level) < LEVELS.index(self.min_level):
            return False
        return not self.prefixes or ev.kind.startswith(self.prefixes)

    def write(self, ev: Event) -> None:      # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        pass


class JsonlHook(Hook):
    """One JSON object per line on a stream. The default, and what a container should use."""

    name = "jsonl"

    def __init__(self, stream=None, min_level: str = "debug", prefixes=()):
        self.stream = stream or sys.stdout
        self.min_level = min_level
        self.prefixes = tuple(prefixes)

    def write(self, ev: Event) -> None:
        self.stream.write(ev.as_json() + "\n")
        self.stream.flush()


class TextHook(Hook):
    """Readable lines, for a person watching a terminal."""

    name = "text"

    def __init__(self, stream=None, min_level: str = "info", prefixes=()):
        self.stream = stream or sys.stderr
        self.min_level = min_level
        self.prefixes = tuple(prefixes)

    def write(self, ev: Event) -> None:
        self.stream.write(ev.line() + "\n")
        self.stream.flush()


class FileHook(Hook):
    """Append JSON lines to a file, reopening if it is rotated out from under us."""

    name = "file"

    def __init__(self, path: str | Path, min_level: str = "debug", prefixes=()):
        self.path = Path(path)
        self.min_level = min_level
        self.prefixes = tuple(prefixes)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)

    def write(self, ev: Event) -> None:
        if not self.path.exists():      # logrotate moved it; start a new one
            self._fh.close()
            self._fh = self.path.open("a", buffering=1)
        self._fh.write(ev.as_json() + "\n")

    def close(self) -> None:
        self._fh.close()


class RingHook(Hook):
    """Keep the last N events in memory, so the API can serve them at `/v1/logs`.

    A ring buffer rather than a list: an unbounded in-process log is a memory leak with a
    schedule, and the only question is whether it fires before the pod is replaced.
    """

    name = "ring"

    def __init__(self, capacity: int = 512, min_level: str = "debug", prefixes=()):
        self.events: deque[Event] = deque(maxlen=capacity)
        self.min_level = min_level
        self.prefixes = tuple(prefixes)
        self._lock = threading.Lock()

    def write(self, ev: Event) -> None:
        with self._lock:
            self.events.append(ev)

    def recent(self, n: int = 100, kind: str | None = None, run: str | None = None) -> list[dict]:
        with self._lock:
            evs = list(self.events)
        if kind:
            evs = [e for e in evs if e.kind.startswith(kind)]
        if run:
            evs = [e for e in evs if e.run == run]
        return [e.as_dict() for e in evs[-n:]]


class CounterHook(Hook):
    """Count events by kind. What `/metrics` exports without keeping any bodies."""

    name = "counter"

    def __init__(self):
        self.counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def write(self, ev: Event) -> None:
        with self._lock:
            self.counts[ev.kind] = self.counts.get(ev.kind, 0) + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counts)


class WebhookHook(Hook):
    """POST each event to a URL. Always wrap this in `Async` — `from_spec` does.

    Standard library only (`urllib`), because adding `requests` to ship a log line would
    contradict the package's one dependency claim.
    """

    name = "webhook"

    def __init__(self, url: str, min_level: str = "info", prefixes=(), timeout: float = 3.0,
                 headers: dict | None = None):
        self.url = url
        self.min_level = min_level
        self.prefixes = tuple(prefixes)
        self.timeout = timeout
        self.headers = headers or {}

    def write(self, ev: Event) -> None:
        import urllib.request
        req = urllib.request.Request(
            self.url, data=ev.as_json().encode(), method="POST",
            headers={"Content-Type": "application/json", **self.headers})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - operator's URL
            r.read(1)


class CallableHook(Hook):
    """Any `f(Event) -> None`, so a notebook or a test can subscribe a lambda."""

    name = "callable"

    def __init__(self, fn: Callable[[Event], None], name: str = "callable",
                 min_level: str = "debug", prefixes=()):
        self.fn = fn
        self.name = name
        self.min_level = min_level
        self.prefixes = tuple(prefixes)

    def write(self, ev: Event) -> None:
        self.fn(ev)


class Async(Hook):
    """Run another hook on a worker thread, through a bounded queue that drops when full.

    This is the piece that makes a network hook safe to put in a request path. The emitting
    thread does a `put_nowait` and moves on; if the consumer cannot keep up the event is dropped
    and `dropped` is incremented. Combined with the monotonic `seq` on every event, a downstream
    consumer can see exactly that it missed something, which is strictly better than an emitter
    that waits on a dead collector.
    """

    name = "async"

    def __init__(self, inner: Hook, maxsize: int = 1000):
        self.inner = inner
        self.name = f"async({inner.name})"
        self.min_level = inner.min_level
        self.prefixes = inner.prefixes
        self.q: queue.Queue[Event | None] = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name=f"sk-hook-{inner.name}")
        self._thread.start()

    def write(self, ev: Event) -> None:
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> None:
        while True:
            ev = self.q.get()
            if ev is None:
                return
            try:
                self.inner.write(ev)
            except Exception:  # noqa: BLE001 - a sink's failure is not the program's failure
                pass

    def flush(self, timeout: float = 2.0) -> bool:
        """Wait for the queue to drain. Returns False on timeout rather than hanging."""
        deadline = time.time() + timeout
        while not self.q.empty() and time.time() < deadline:
            time.sleep(0.01)
        return self.q.empty()

    def close(self) -> None:
        self.flush()
        self.q.put(None)
        self.inner.close()


# ----------------------------------------------------------------------------------- bus
class EventBus:
    """Fan one event out to every subscribed hook, and survive all of them."""

    MUTE_AFTER = 5

    def __init__(self) -> None:
        self.hooks: list[Hook] = []
        self.errors: dict[str, int] = {}
        self.muted: set[str] = set()
        # The spec strings already installed, so configuring twice from two places does not
        # double every log line. It happened immediately: the CLI reads $SERVINGKIT_LOG_HOOKS
        # before dispatching and `serve()` read it again, and the container duly emitted each
        # event twice to each sink. An operator calling configure() twice deserves the same
        # protection as a caller who did it by accident.
        self.installed: set[str] = set()
        self._seq = 0
        self._lock = threading.Lock()
        self._run: str | None = None

    # ------------------------------------------------------------ subscription
    def subscribe(self, hook: Hook) -> Hook:
        with self._lock:
            self.hooks.append(hook)
        return hook

    def unsubscribe(self, hook: Hook) -> None:
        with self._lock:
            if hook in self.hooks:
                self.hooks.remove(hook)
        hook.close()

    def clear(self) -> None:
        with self._lock:
            hooks, self.hooks = self.hooks, []
            self.errors.clear()
            self.muted.clear()
            self.installed.clear()
        for h in hooks:
            h.close()

    # -------------------------------------------------------------------- runs
    def run_id(self, prefix: str = "run") -> str:
        return f"{prefix}-{int(time.time() * 1000) % 10**9:09d}"

    class _Run:
        def __init__(self, bus: "EventBus", rid: str):
            self.bus, self.rid, self.prev = bus, rid, None

        def __enter__(self) -> str:
            self.prev = self.bus._run
            self.bus._run = self.rid
            return self.rid

        def __exit__(self, *exc) -> None:
            self.bus._run = self.prev

    def run(self, rid: str | None = None, prefix: str = "run") -> "_Run":
        """Tag every event emitted inside the block with one run id."""
        return EventBus._Run(self, rid or self.run_id(prefix))

    # -------------------------------------------------------------------- emit
    def emit(self, kind: str, /, level: str = "info", **data) -> Event:
        """Emit one event. `kind` is positional-only, and that is not a style preference.

        Event payloads are arbitrary dicts assembled by callers, and `kind` is an ordinary word:
        a pipeline stage whose detail includes the workload's `kind`, or a request handler logging
        the `kind` of workload asked for, both collide with the parameter and raise
        `TypeError: got multiple values for argument 'kind'` — at which point a logging call has
        taken down the thing it was logging. Marking it positional-only makes `kind=` in the
        payload land in `data`, where it belongs.
        """
        with self._lock:
            self._seq += 1
            ev = Event(seq=self._seq, ts=time.time(), kind=kind, level=level,
                       run=self._run, data=data)
            hooks = list(self.hooks)
        for h in hooks:
            if h.name in self.muted:
                continue
            try:
                if h.wants(ev):
                    h.write(ev)
            except Exception as exc:  # noqa: BLE001 - see the module docstring
                n = self.errors.get(h.name, 0) + 1
                self.errors[h.name] = n
                if n >= self.MUTE_AFTER:
                    self.muted.add(h.name)
                    sys.stderr.write(
                        f"servingkit: log hook {h.name!r} muted after {n} failures "
                        f"(last: {type(exc).__name__}: {exc})\n")
        return ev

    def health(self) -> dict:
        """Whether the logging itself is working. A broken collector should be visible."""
        return dict(
            hooks=[dict(name=h.name, min_level=h.min_level, prefixes=list(h.prefixes),
                        muted=h.name in self.muted, errors=self.errors.get(h.name, 0),
                        dropped=getattr(h, "dropped", 0)) for h in self.hooks],
            events=self._seq, muted=sorted(self.muted), errors=dict(self.errors),
            dropped=sum(getattr(h, "dropped", 0) for h in self.hooks),
        )

    def flush(self, timeout: float = 2.0) -> bool:
        return all(h.flush(timeout) for h in self.hooks if isinstance(h, Async))


BUS = EventBus()


def emit(kind: str, /, level: str = "info", **data) -> Event:
    """Module-level shorthand, so callers do not have to import the bus object."""
    return BUS.emit(kind, level=level, **data)


# ------------------------------------------------------------------------------- wiring
def from_spec(spec: str) -> Hook:
    """Build a hook from a compact string, so the environment can configure logging.

        jsonl            JSON lines on stdout
        jsonl:-          the same, explicitly
        text             readable lines on stderr
        file:/path.jsonl JSON lines appended to a file
        ring:512         keep the last 512 in memory for /v1/logs
        counter          count by kind, for /metrics
        webhook:URL      POST each one, on a worker thread, dropping under backpressure
        <any>@warn       only at this level or above
        <any>#agent.     only kinds with this prefix
        plugin:mod:attr  anything importable that is a Hook or an f(Event)
    """
    body, level, prefixes = spec.strip(), "debug", ()
    if "#" in body:
        body, _, pre = body.partition("#")
        prefixes = tuple(p for p in pre.split("+") if p)
    if "@" in body and not body.startswith(("webhook:", "plugin:")):
        body, _, level = body.partition("@")
    elif "@" in body.split("://")[-1] and body.startswith("webhook:"):
        pass                                        # an @ in a URL is not a level
    kind, _, arg = body.partition(":")
    kind = kind.strip().lower()

    if kind in ("jsonl", "json"):
        return JsonlHook(None if arg in ("", "-") else open(arg, "a", buffering=1),  # noqa: SIM115
                         min_level=level, prefixes=prefixes)
    if kind == "text":
        return TextHook(min_level=level or "info", prefixes=prefixes)
    if kind == "file":
        if not arg:
            raise ValueError("file hook needs a path: file:/var/log/servingkit.jsonl")
        return FileHook(arg, min_level=level, prefixes=prefixes)
    if kind == "ring":
        return RingHook(int(arg or 512), min_level=level, prefixes=prefixes)
    if kind == "counter":
        return CounterHook()
    if kind == "webhook":
        if "@" in arg and arg.rsplit("@", 1)[-1] in LEVELS:
            arg, _, level = arg.rpartition("@")
        if not arg:
            raise ValueError("webhook hook needs a URL")
        return Async(WebhookHook(arg, min_level=level if level in LEVELS else "info",
                                 prefixes=prefixes))
    if kind == "plugin":
        return _plugin(arg, level, prefixes)
    raise ValueError(f"unknown log hook {spec!r}")


def _plugin(target: str, level: str, prefixes) -> Hook:
    """`module:attr`, resolved by import. The extension point with no registry.

    The attribute may be a Hook, a Hook subclass, or a plain callable — all three are things
    somebody would reasonably write, and insisting on one of them would mean a wrapper class in
    every deployment that wants to ship its own log line.
    """
    import importlib
    mod_name, _, attr = target.partition(":")
    if not mod_name or not attr:
        raise ValueError(f"plugin hook wants module:attr, got {target!r}")
    obj = getattr(importlib.import_module(mod_name), attr)
    if isinstance(obj, Hook):
        return obj
    if isinstance(obj, type) and issubclass(obj, Hook):
        return obj()
    if callable(obj):
        return CallableHook(obj, name=f"plugin({target})", min_level=level, prefixes=prefixes)
    raise TypeError(f"{target} is neither a Hook nor callable")


def configure(specs: str | list[str] | None = None, *, default: str | None = None,
              bus: EventBus | None = None, force: bool = False) -> list[Hook]:
    """Subscribe the hooks named by `specs`, or by `$SERVINGKIT_LOG_HOOKS`.

    A bad spec is a warning, not a crash. Refusing to start a service because one log sink was
    misspelled trades a degraded deployment for no deployment.

    Calling this twice with the same spec installs it once: two entry points that both read the
    environment is a normal shape for a program that is both a CLI and a server, and it should
    not mean two copies of every log line. `force=True` if you really want another one.
    """
    bus = bus or BUS
    raw = specs if specs is not None else os.environ.get("SERVINGKIT_LOG_HOOKS", default or "")
    if isinstance(raw, str):
        # Split on commas, but not inside a URL's query string.
        items = [s for s in (x.strip() for x in raw.split(",")) if s]
    else:
        items = list(raw)
    out = []
    for s in items:
        if not force and s in bus.installed:
            continue
        try:
            out.append(bus.subscribe(from_spec(s)))
            bus.installed.add(s)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"servingkit: ignoring log hook {s!r}: "
                             f"{type(exc).__name__}: {exc}\n")
    return out


__all__ = ["BUS", "Event", "EventBus", "Hook", "JsonlHook", "TextHook", "FileHook", "RingHook",
           "CounterHook", "WebhookHook", "CallableHook", "Async", "configure", "from_spec",
           "emit", "LEVELS"]
