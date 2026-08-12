#!/usr/bin/env python3
"""gpu_bench.py — measure GPU code without lying to yourself.

    from tools.gpu_bench import Bench
    b = Bench()                              # queries and *measures* the device
    r = b.time(lambda: y.copy_(x), "copy", nbytes=2 * x.nbytes)
    b.table([r])

Run it directly for a self-test and a report on whatever hardware is present:

    python tools/gpu_bench.py

Why this file exists
--------------------
Almost every wrong GPU benchmark is wrong in one of six ways, and all six are mechanical:

  1. **Timed with the host clock.** A kernel launch is asynchronous. `t0 = time(); k(); t1 =
     time()` measures how long it took to *enqueue* the kernel — typically 5-10 us regardless
     of whether the kernel runs for 1 us or 1 second. `Bench.wall_time()` reproduces this
     mistake on purpose so you can see the size of it.

  2. **No warmup.** The first call pays for context creation, module load, JIT, autotuning and
     an allocator that has not yet cached anything. On a fresh CUDA context that is tens of
     milliseconds against a kernel that runs for tens of microseconds.

  3. **Hot cache.** Run the same kernel 100 times on the same 32 MB input and, after the first
     iteration, it is served from a 50 MB L2. A memory-bound kernel then reports 3-5x the
     bandwidth the card actually has. `flush_l2=True` writes an L2-sized buffer between reps.

  4. **Mean over few reps.** GPU timing distributions are a tight mode with a right tail from
     clock throttling, other tenants, and interrupts. The mean chases the tail. Median and MAD
     do not, and the tail is reported separately as p95 rather than being averaged into the
     headline.

  5. **Launch overhead inside the measurement.** For kernels of a few microseconds, the ~5 us
     launch is most of the number. `use_graph=True` captures the work into a CUDA graph and
     replays it, which is also what a serving engine does in production — so this is not a
     measurement trick, it is measuring the thing that will actually run.

  6. **No denominator.** 400 GB/s is excellent on a T4 and a catastrophe on an H100. Every
     result here is divided by a ceiling, and the ceiling is *measured on the same machine in
     the same process*, not read off a spec sheet — vendor peak numbers assume clocks no real
     workload sustains.

No GPU present: everything falls back to `time.perf_counter`, and every result is labelled
`wall` instead of `cuda-events` so a CPU-only run cannot be mistaken for a GPU one.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - the notebooks install it
    torch = None


# Vendor peak numbers, for the "what fraction of the sticker am I getting?" column only.
# Measured ceilings are what the roofline actually uses.
SPEC_PEAKS = {
    # name substring     : (HBM GB/s, fp32 TFLOP/s, fp16 tensor-core TFLOP/s dense)
    "H200":               (4800, 67, 990),
    "H100 80GB HBM3":     (3350, 67, 990),
    "H100":               (2000, 67, 756),
    "A100":               (2039, 19.5, 312),
    "L40":                (864, 91, 362),
    "L4":                 (300, 30, 121),
    "A10":                (600, 31, 125),
    "V100":               (900, 15.7, 125),
    "T4":                 (320, 8.1, 65),
    "RTX 4090":           (1008, 82, 165),
    "RTX 3090":           (936, 35, 71),
}


def _spec_for(name: str):
    for key, vals in SPEC_PEAKS.items():
        if key.lower() in name.lower():
            return vals
    return None


@dataclass
class Result:
    name: str
    samples_ms: list[float]
    nbytes: float = 0.0
    flops: float = 0.0
    timer: str = "cuda-events"
    note: str = ""

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def mad_ms(self) -> float:
        """Median absolute deviation — a spread that one slow rep cannot move."""
        m = self.median_ms
        return statistics.median([abs(s - m) for s in self.samples_ms])

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def p95_ms(self) -> float:
        s = sorted(self.samples_ms)
        return s[min(len(s) - 1, int(0.95 * (len(s) - 1)))]

    @property
    def ci95_ms(self) -> tuple[float, float]:
        """Bootstrap 95% interval for the median. Reported because 'the median of 50 reps'
        is itself an estimate, and quoting four significant figures for it is a fiction."""
        s = sorted(self.samples_ms)
        n = len(s)
        if n < 8:
            return (s[0], s[-1])
        # Normal approximation to the binomial order statistics of the median.
        half = 1.96 * math.sqrt(n) / 2.0
        lo = max(0, int(math.floor(n / 2 - half)))
        hi = min(n - 1, int(math.ceil(n / 2 + half)))
        return (s[lo], s[hi])

    @property
    def gbps(self) -> float:
        return self.nbytes / (self.median_ms / 1e3) / 1e9 if self.median_ms else 0.0

    @property
    def tflops(self) -> float:
        return self.flops / (self.median_ms / 1e3) / 1e12 if self.median_ms else 0.0

    @property
    def intensity(self) -> float:
        return self.flops / self.nbytes if self.nbytes else float("inf")


@dataclass
class Device:
    name: str = "cpu"
    has_cuda: bool = False
    sms: int = 0
    l2_bytes: int = 0
    total_gb: float = 0.0
    measured_gbps: float = 0.0
    measured_tflops_fp32: float = 0.0
    measured_tflops_fp16: float = 0.0
    spec: tuple | None = None

    @property
    def ridge_fp32(self) -> float:
        """FLOP/byte above which a kernel is compute-bound rather than memory-bound."""
        if not self.measured_gbps:
            return float("nan")
        return self.measured_tflops_fp32 * 1e12 / (self.measured_gbps * 1e9)

    @property
    def ridge_fp16(self) -> float:
        if not self.measured_gbps:
            return float("nan")
        return self.measured_tflops_fp16 * 1e12 / (self.measured_gbps * 1e9)


class Bench:
    def __init__(self, warmup: int = 10, reps: int = 50, flush_l2: bool = True,
                 measure_peaks: bool = True, quiet: bool = False):
        self.warmup = warmup
        self.reps = reps
        self.flush_l2 = flush_l2
        self.quiet = quiet
        self.dev = Device()
        self._flush_buf = None

        if torch is not None and torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            self.dev = Device(
                name=p.name,
                has_cuda=True,
                sms=p.multi_processor_count,
                l2_bytes=int(getattr(p, "L2_cache_size", 0) or 40 * 1024 * 1024),
                total_gb=p.total_memory / 1e9,
                spec=_spec_for(p.name),
            )
            if self.flush_l2:
                n = max(1, self.dev.l2_bytes * 3 // 4)
                self._flush_buf = torch.empty(n, dtype=torch.float32, device="cuda")
            if measure_peaks:
                self._measure_ceilings()

    # -- the ceilings -------------------------------------------------------------------
    def _measure_ceilings(self) -> None:
        """Measure this machine's own achievable bandwidth and FLOP rate.

        A roofline drawn against vendor peak flatters every kernel by the 10-25% that peak
        assumes and no real workload sustains. Drawn against what a *saturating* kernel gets
        on this card, in this process, at this clock, a well-written kernel lands near 100%
        and a badly-written one has nowhere to hide.
        """
        d = "cuda"
        # Bandwidth: a large copy is the canonical bandwidth-saturating kernel. Sized well past
        # L2 so it cannot be served from cache.
        n = max(1 << 24, self.dev.l2_bytes * 4 // 4)
        x = torch.empty(n, dtype=torch.float32, device=d)
        y = torch.empty_like(x)
        r = self.time(lambda: y.copy_(x), "peak-bw probe", nbytes=2 * x.numel() * 4,
                      reps=20, warmup=5, _internal=True)
        self.dev.measured_gbps = r.gbps
        del x, y

        # FLOPs: a large square matmul. fp32 and fp16 separately — on any card since Volta
        # they differ by 8-20x, and quoting the wrong one puts the ridge point in the wrong
        # place by the same factor.
        for dtype, attr in ((torch.float32, "measured_tflops_fp32"),
                            (torch.float16, "measured_tflops_fp16")):
            try:
                m = 4096
                a = torch.randn(m, m, dtype=dtype, device=d)
                b = torch.randn(m, m, dtype=dtype, device=d)
                r = self.time(lambda: torch.mm(a, b), f"peak-{dtype}", flops=2.0 * m ** 3,
                              reps=10, warmup=5, _internal=True)
                setattr(self.dev, attr, r.tflops)
                del a, b
            except RuntimeError:
                pass
        torch.cuda.empty_cache()

    # -- timing -------------------------------------------------------------------------
    def _flush(self) -> None:
        if self._flush_buf is not None:
            self._flush_buf.zero_()

    def time(self, fn: Callable[[], object], name: str = "", nbytes: float = 0.0,
             flops: float = 0.0, reps: int | None = None, warmup: int | None = None,
             use_graph: bool = False, _internal: bool = False) -> Result:
        """Time `fn` properly. `fn` should enqueue work and nothing else."""
        reps = self.reps if reps is None else reps
        warmup = self.warmup if warmup is None else warmup

        if not self.dev.has_cuda:
            for _ in range(warmup):
                fn()
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1e3)
            return Result(name or "fn", samples, nbytes, flops, timer="wall (no CUDA)")

        if use_graph:
            return self._time_graph(fn, name, nbytes, flops, reps, warmup)

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(reps):
            self._flush()
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
        return Result(name or "fn", samples, nbytes, flops, timer="cuda-events")

    def _time_graph(self, fn, name, nbytes, flops, reps, warmup) -> Result:
        """Capture into a CUDA graph and replay.

        Removes per-launch CPU cost from the measurement, which for kernels under ~20 us is
        most of it. Capture happens on a side stream because the legacy default stream cannot
        be captured. Anything that synchronizes, allocates, or touches the CPU inside `fn`
        will fail capture — which is itself informative, since the same restriction applies
        when a serving engine tries to graph its decode step.
        """
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()

        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(reps):
            self._flush()
            start.record()
            g.replay()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
        return Result(name or "fn", samples, nbytes, flops, timer="cuda-graph",
                      note="launch overhead removed")

    def wall_time(self, fn: Callable[[], object], name: str = "", reps: int = 50) -> Result:
        """The wrong way, kept so the notebooks can show how wrong it is.

        No warmup, no sync, host clock around an asynchronous launch. On a GPU this returns
        the enqueue cost; the kernel is usually still running when the timer stops.
        """
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
        return Result(name or "fn", samples, timer="wall clock, no sync",
                      note="measures enqueue, not execution")

    # -- reporting ----------------------------------------------------------------------
    def describe(self) -> str:
        d = self.dev
        if not d.has_cuda:
            return ("device : CPU only (no CUDA). Timings below are wall clock and are not\n"
                    "         comparable to GPU numbers; they are here so the code runs.")
        out = [f"device : {d.name} — {d.sms} SMs, {d.l2_bytes/1048576:.0f} MB L2, "
               f"{d.total_gb:.0f} GB"]
        out.append(f"measured ceilings on THIS card, in THIS process:")
        out.append(f"  bandwidth      {d.measured_gbps:8.0f} GB/s   (large device-to-device copy)")
        out.append(f"  fp32 matmul    {d.measured_tflops_fp32:8.1f} TFLOP/s")
        out.append(f"  fp16 matmul    {d.measured_tflops_fp16:8.1f} TFLOP/s  (tensor cores)")
        if d.spec:
            bw, f32, f16 = d.spec
            out.append(f"  vs vendor peak {100*d.measured_gbps/bw:7.0f}% BW, "
                       f"{100*d.measured_tflops_fp32/f32:.0f}% fp32, "
                       f"{100*d.measured_tflops_fp16/f16:.0f}% fp16")
        out.append(f"  ridge point    {d.ridge_fp32:8.1f} FLOP/byte fp32, "
                   f"{d.ridge_fp16:.1f} fp16")
        out.append("  below the ridge a kernel is memory-bound and only fewer bytes will help.")
        return "\n".join(out)

    def table(self, results: Sequence[Result], baseline: int | None = None) -> str:
        hdr = (f"{'kernel':<28}{'median ms':>11}{'±MAD':>9}{'p95':>9}"
               f"{'GB/s':>9}{'TFLOP/s':>9}{'% ceiling':>11}  {'timer':<14}")
        lines = [hdr, "-" * len(hdr)]
        base = results[baseline].median_ms if baseline is not None else None
        for r in results:
            pct = ""
            if self.dev.has_cuda and (r.nbytes or r.flops):
                fb = r.gbps / self.dev.measured_gbps if self.dev.measured_gbps else 0
                fc = (r.tflops / self.dev.measured_tflops_fp32
                      if self.dev.measured_tflops_fp32 else 0)
                pct = (f"{100*fb:.0f}% BW" if fb >= fc else f"{100*fc:.0f}% FLOP")
            line = (f"{r.name:<28}{r.median_ms:>11.4f}{r.mad_ms:>9.4f}{r.p95_ms:>9.4f}"
                    f"{r.gbps:>9.1f}{r.tflops:>9.2f}{pct:>11}  {r.timer:<14}")
            if base is not None and r.median_ms:
                line += f"  {base / r.median_ms:.2f}x"
            if r.note:
                line += f"  {r.note}"
            lines.append(line)
        return "\n".join(lines)

    def roofline(self, r: Result) -> str:
        """Name the resource this kernel is up against, and what the ceiling implies."""
        d = self.dev
        if not d.has_cuda or not r.nbytes:
            return ""
        ridge = d.ridge_fp32
        bound = "memory" if r.intensity < ridge else "compute"
        attainable = min(d.measured_gbps * 1e9 * r.intensity, d.measured_tflops_fp32 * 1e12)
        achieved = r.flops / (r.median_ms / 1e3) if r.flops else 0.0
        frac = achieved / attainable if attainable else 0.0
        msg = [f"{r.name}: {r.intensity:.2f} FLOP/byte vs ridge {ridge:.0f} -> {bound}-bound"]
        if r.flops:
            msg.append(f"  attainable here: {attainable/1e12:.2f} TFLOP/s; "
                       f"achieved {achieved/1e12:.2f} ({100*frac:.0f}%)")
        if bound == "memory":
            msg.append("  more FLOPs are free; fewer bytes is the only lever.")
        return "\n".join(msg)


def _self_test() -> int:
    b = Bench()
    print(b.describe())
    print()
    if not b.dev.has_cuda:
        # Still exercise every code path so CI covers them.
        r = b.time(lambda: sum(range(10000)), "python sum", reps=5, warmup=1)
        print(b.table([r]))
        assert r.median_ms > 0 and r.timer.startswith("wall")
        assert r.ci95_ms[0] <= r.median_ms <= r.ci95_ms[1]
        print("\nCPU fallback paths OK (no GPU present, so no GPU claims are made).")
        return 0

    n = 1 << 26
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)
    nbytes = 2 * x.numel() * x.element_size()

    results = [
        b.wall_time(lambda: y.copy_(x), "copy, wall clock"),
        b.time(lambda: y.copy_(x), "copy, cuda events", nbytes=nbytes),
        b.time(lambda: y.copy_(x), "copy, no L2 flush", nbytes=nbytes),
    ]
    saved, b.flush_l2 = b.flush_l2, False
    buf, b._flush_buf = b._flush_buf, None
    small = torch.randn(1 << 18, device="cuda")
    small_out = torch.empty_like(small)
    results.append(b.time(lambda: small_out.copy_(small), "small copy, hot L2",
                          nbytes=2 * small.numel() * 4))
    b.flush_l2, b._flush_buf = saved, buf
    results.append(b.time(lambda: small_out.copy_(small), "small copy, cold L2",
                          nbytes=2 * small.numel() * 4))

    m = 2048
    A = torch.randn(m, m, device="cuda")
    B = torch.randn(m, m, device="cuda")
    results.append(b.time(lambda: torch.mm(A, B), "sgemm 2048", flops=2.0 * m ** 3,
                          nbytes=3.0 * m * m * 4))
    print(b.table(results))
    print()
    print(b.roofline(results[1]))
    print(b.roofline(results[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
