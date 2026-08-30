"""Scorecards. One for a terminal, one for a pull request, one for a machine."""
from __future__ import annotations

import json
from pathlib import Path

from .evaluate import CaseResult

DIMS = ["build", "correctness", "variants", "determinism", "mutation", "sanity",
        "efficiency"]


def _cell(r: CaseResult, dim: str) -> str:
    d = r.dims.get(dim)
    if d is None:
        return "  -  "
    if d.skipped:
        return "  -  "
    if d.total == 0:
        return "  -  "
    return f"{d.passed}/{d.total}" + ("" if d.ok() else " !")


def terminal(results: list[CaseResult], suite_name: str, device: str) -> str:
    w = 26
    head = f"{'kernel':<{w}}" + "".join(f"{d[:12]:>14}" for d in DIMS) + f"{'score':>9}"
    lines = [f"kernelbench — {suite_name}", f"device: {device}", "", head, "-" * len(head)]
    for r in results:
        row = f"{r.case.name:<{w}}" + "".join(f"{_cell(r, d):>14}" for d in DIMS)
        lines.append(row + f"{r.score:>8.0%}")
    lines.append("-" * len(head))

    agg = {d: [0, 0] for d in DIMS}
    for r in results:
        for d in DIMS:
            dim = r.dims.get(d)
            if dim and not dim.skipped:
                agg[d][0] += dim.passed
                agg[d][1] += dim.total
    tot = f"{'TOTAL':<{w}}" + "".join(
        f"{(f'{agg[d][0]}/{agg[d][1]}' if agg[d][1] else '-'):>14}" for d in DIMS)
    overall = sum(r.score for r in results) / len(results) if results else 0.0
    lines.append(tot + f"{overall:>8.0%}")

    problems = [(r, d, dim) for r in results for d, dim in r.dims.items() if not dim.ok()]
    if problems:
        lines.append("")
        lines.append(f"{len(problems)} failing dimension(s):")
        for r, d, dim in problems:
            lines.append(f"  {r.case.name} / {d}")
            for det in dim.detail:
                for i, l in enumerate(det.split("\n")[:8]):
                    lines.append(f"      {l}")
    else:
        # Say what was actually established, not just "ok".
        muts = sum(r.dims["mutation"].total for r in results if "mutation" in r.dims
                   and not r.dims["mutation"].skipped)
        lines.append("")
        lines.append(f"all dimensions pass; {muts} injected bugs were caught, so the "
                     f"correctness checks are known to be capable of failing")
    return "\n".join(lines)


def markdown(results: list[CaseResult], suite_name: str, device: str) -> str:
    out = [f"## kernelbench — {suite_name}", "",
           f"`{device}`", "",
           "| kernel | " + " | ".join(DIMS) + " | score |",
           "|---" * (len(DIMS) + 2) + "|"]
    for r in results:
        cells = " | ".join(_cell(r, d).strip() for d in DIMS)
        out.append(f"| `{r.case.name}` | {cells} | {r.score:.0%} |")
    overall = sum(r.score for r in results) / len(results) if results else 0.0
    out.append(f"| **total** |" + " |" * len(DIMS) + f" **{overall:.0%}** |")
    out += ["", "Dimensions: **build** compiles under every available toolchain · "
            "**correctness** every variant matches the program's own CPU reference · "
            "**variants** the expected number of implementations is present · "
            "**determinism** results are stable under shuffled block order, and the variants "
            "declared order-dependent really are · **mutation** every injected bug is caught · "
            "**efficiency** the best variant reaches a declared fraction of the measured "
            "hardware ceiling (GPU only)."]
    fails = [(r, d, dim) for r in results for d, dim in r.dims.items() if not dim.ok()]
    if fails:
        out += ["", "### Failures", ""]
        for r, d, dim in fails:
            out.append(f"- **`{r.case.name}` / {d}**")
            for det in dim.detail:
                out.append(f"  - {det.splitlines()[0]}")
    return "\n".join(out)


def to_json(results: list[CaseResult], suite_name: str, device: str) -> str:
    payload = {
        "suite": suite_name,
        "device": device,
        "overall": (sum(r.score for r in results) / len(results)) if results else 0.0,
        "kernels": [
            {
                "name": r.case.name,
                "source": r.case.source,
                "title": r.case.title,
                "topic": r.case.topic,
                "score": r.score,
                "seconds": round(r.seconds, 2),
                "dimensions": {
                    d: {"passed": dim.passed, "total": dim.total,
                        "skipped": dim.skipped, "detail": dim.detail}
                    for d, dim in r.dims.items()
                },
                "variants": (r.kb or {}).get("variants", []),
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2)


def write(results, suite_name, device, json_path=None, md_path=None,
          html_path=None) -> None:
    if json_path:
        Path(json_path).write_text(to_json(results, suite_name, device))
    if md_path:
        Path(md_path).write_text(markdown(results, suite_name, device))
    if html_path:
        Path(html_path).write_text(html(results, suite_name, device), encoding="utf-8")


# ---------------------------------------------------------------------------------------
# The visual scorecard. Self-contained: no CDN, no build step, opens from a file:// URL and
# survives being attached to a CI run or dropped into a PR.
# ---------------------------------------------------------------------------------------
CSS = """
:root{--bg:#faf9f7;--panel:#fff;--ink:#1a1a1a;--ink2:#4a4a4a;--ink3:#8a8580;
 --line:#e2ded8;--line2:#f0ece6;--ok:#2a8a5c;--bad:#c0392b;--warn:#b8860b;--na:#c9c4bd;
 --accent:#8a5a2b;--shadow:0 1px 2px rgba(0,0,0,.05);
 --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;
 --serif:Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --bg:#171614;--panel:#201e1b;--ink:#eeeae4;--ink2:#bdb7ae;--ink3:#8a8177;
 --line:#332f2a;--line2:#282521;--ok:#4fb383;--bad:#e8705f;--warn:#d9a441;--na:#4a453e;
 --accent:#d9a06a;--shadow:0 1px 2px rgba(0,0,0,.3);}}
:root[data-theme="dark"]{--bg:#171614;--panel:#201e1b;--ink:#eeeae4;--ink2:#bdb7ae;
 --ink3:#8a8177;--line:#332f2a;--line2:#282521;--ok:#4fb383;--bad:#e8705f;--warn:#d9a441;
 --na:#4a453e;--accent:#d9a06a;--shadow:0 1px 2px rgba(0,0,0,.3);}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--serif);margin:0;
 line-height:1.55;font-size:16px}
.wrap{max-width:1080px;margin:0 auto;padding:38px 24px 70px}
h1{font-family:var(--mono);font-size:23px;font-weight:600;letter-spacing:-.02em;margin:0 0 4px}
h2{font-family:var(--mono);font-size:15px;font-weight:600;letter-spacing:.02em;
 text-transform:uppercase;color:var(--ink3);margin:38px 0 12px}
.sub{color:var(--ink2);margin:0 0 22px;font-size:15px}
code,.mono{font-family:var(--mono)}
.top{display:flex;gap:26px;align-items:center;flex-wrap:wrap;
 border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:20px 24px;
 box-shadow:var(--shadow)}
.ring{--pct:0;width:96px;height:96px;border-radius:50%;flex:0 0 auto;display:grid;
 place-items:center;position:relative;
 background:conic-gradient(var(--ok) calc(var(--pct)*1%),var(--line2) 0)}
.ring.bad{background:conic-gradient(var(--bad) calc(var(--pct)*1%),var(--line2) 0)}
.ring span{width:74px;height:74px;border-radius:50%;background:var(--panel);display:grid;
 place-items:center;font-family:var(--mono);font-size:21px;font-weight:600}
.facts{display:flex;gap:30px;flex-wrap:wrap}
.fact b{display:block;font-family:var(--mono);font-size:21px;font-weight:600;
 font-variant-numeric:tabular-nums}
.fact span{font-size:13.5px;color:var(--ink3)}
table{border-collapse:collapse;width:100%;font-size:14px}
th{font-family:var(--mono);font-size:11px;letter-spacing:.07em;text-transform:uppercase;
 color:var(--ink3);text-align:center;padding:8px 6px;font-weight:600;border-bottom:1px solid var(--line)}
th.l,td.l{text-align:left}
td{padding:7px 6px;text-align:center;border-bottom:1px solid var(--line2);
 font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums}
tr:hover td{background:var(--line2)}
.pill{display:inline-block;min-width:44px;padding:2px 7px;border-radius:3px;font-size:11.5px;
 font-weight:600}
.pill.ok{background:color-mix(in srgb,var(--ok) 16%,transparent);color:var(--ok)}
.pill.bad{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.pill.na{color:var(--na)}
.card{border:1px solid var(--line);background:var(--panel);border-radius:6px;margin:12px 0;
 box-shadow:var(--shadow);overflow:hidden}
.card>summary{cursor:pointer;padding:13px 18px;display:flex;gap:14px;align-items:baseline;
 list-style:none}
.card>summary::-webkit-details-marker{display:none}
.card>summary:hover{background:var(--line2)}
.card .nm{font-family:var(--mono);font-size:14.5px;font-weight:600;color:var(--accent)}
.card .ti{color:var(--ink2);font-size:14.5px;flex:1}
.card .sc{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums}
.body{padding:4px 18px 18px;border-top:1px solid var(--line2)}
.bar{height:7px;border-radius:2px;background:var(--accent);opacity:.75;display:inline-block;
 vertical-align:middle}
.note{color:var(--ink3);font-size:12px}
ul.det{margin:8px 0 0;padding-left:20px;font-size:13.5px;color:var(--ink2)}
ul.det li{margin:3px 0}
.legend{font-size:14px;color:var(--ink2)}
.legend b{font-family:var(--mono);font-size:12.5px;color:var(--ink)}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);
 font-size:13.5px;color:var(--ink3)}
"""

_DIM_HELP = {
    "build": "compiles under every toolchain present — nvcc when there is one, and always g++ "
             "against the CPU shim",
    "correctness": "every variant matches the program's own reference, computed in double, on "
                   "the same run that is being timed",
    "variants": "the suite's expected number of implementations is present, so one cannot "
                "quietly disappear",
    "determinism": "results are bit-stable across shuffled block orders — and the variants "
                   "declared order-dependent really are, checked in both directions",
    "mutation": "a realistic bug is injected into each kernel and the check must catch it. "
                "This is the only dimension that measures the test rather than the code",
    "sanity": "no variant reports more bandwidth or FLOP/s than the hardware physically has "
              "(GPU only)",
    "efficiency": "the best variant reaches the fraction of the measured hardware ceiling the "
                  "suite declares (GPU only)",
}


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def html(results: list[CaseResult], suite_name: str, device: str) -> str:
    overall = sum(r.score for r in results) / len(results) if results else 0.0
    n_var = sum(len((r.kb or {}).get("variants", [])) for r in results)
    n_mut = sum(r.dims["mutation"].total for r in results
                if "mutation" in r.dims and not r.dims["mutation"].skipped)
    n_fail = sum(1 for r in results for d in r.dims.values() if not d.ok())
    timed = any(v.get("timed") for r in results for v in (r.kb or {}).get("variants", []))

    o: list[str] = []
    o.append(f"<title>kernelbench — {_esc(suite_name)}</title>")
    o.append(f"<style>{CSS}</style>")
    o.append('<div class="wrap">')
    o.append(f"<h1>kernelbench</h1>")
    o.append(f'<p class="sub">{_esc(suite_name)} &middot; <span class="mono">'
             f'{_esc(device)}</span></p>')

    ring_cls = "ring" if n_fail == 0 else "ring bad"
    o.append(f'<div class="top"><div class="{ring_cls}" style="--pct:{overall*100:.0f}">'
             f'<span>{overall:.0%}</span></div><div class="facts">')
    for val, lab in ((len(results), "kernels"), (n_var, "implementations checked"),
                     (n_mut, "bugs injected and caught"),
                     (n_fail, "failing dimensions")):
        o.append(f'<div class="fact"><b>{val}</b><span>{lab}</span></div>')
    o.append("</div></div>")

    if not timed:
        o.append('<p class="sub" style="margin-top:16px">No GPU was present, so this run '
                 'checked correctness, determinism and mutation coverage and reported no '
                 'timings at all &mdash; rather than numbers that could be mistaken for GPU '
                 'results.</p>')

    # --- the matrix ---------------------------------------------------------------------
    o.append("<h2>Scorecard</h2><table><thead><tr><th class='l'>kernel</th>")
    for d in DIMS:
        o.append(f"<th>{d}</th>")
    o.append("<th>score</th></tr></thead><tbody>")
    for r in results:
        o.append(f"<tr><td class='l'><b>{_esc(r.case.name)}</b></td>")
        for dname in DIMS:
            dim = r.dims.get(dname)
            if dim is None or dim.skipped or dim.total == 0:
                o.append('<td><span class="pill na">&mdash;</span></td>')
            else:
                cls = "ok" if dim.ok() else "bad"
                o.append(f'<td><span class="pill {cls}">{dim.passed}/{dim.total}</span></td>')
        o.append(f"<td>{r.score:.0%}</td></tr>")
    o.append("</tbody></table>")

    # --- per-kernel detail ---------------------------------------------------------------
    o.append("<h2>Kernels</h2>")
    for r in results:
        variants = (r.kb or {}).get("variants", [])
        slow = max((v["median_ms"] for v in variants if v.get("timed")), default=0.0)
        bad = [d for d in r.dims.values() if not d.ok()]
        o.append(f'<details class="card"{" open" if bad else ""}><summary>'
                 f'<span class="nm">{_esc(r.case.name)}</span>'
                 f'<span class="ti">{_esc(r.case.title)}</span>'
                 f'<span class="sc">{r.score:.0%}</span></summary><div class="body">')
        if variants:
            o.append("<table><thead><tr><th class='l'>implementation</th>")
            if timed:
                o.append("<th>median ms</th><th>relative</th><th>GB/s</th><th>GFLOP/s</th>")
            o.append("<th>max err</th><th>checksum</th><th class='l'>note</th>"
                     "</tr></thead><tbody>")
            for v in variants:
                o.append(f"<tr><td class='l'>{_esc(v['name'])}</td>")
                if timed:
                    w = 130 * (v["median_ms"] / slow) if slow else 0
                    o.append(f"<td>{v['median_ms']:.4f}</td>"
                             f"<td><span class='bar' style='width:{w:.0f}px'></span></td>"
                             f"<td>{v['gbps']:.0f}</td><td>{v['gflops']:.0f}</td>")
                cls = "ok" if v["ok"] else "bad"
                o.append(f"<td><span class='pill {cls}'>{v['err']:.1e}</span></td>"
                         f"<td class='note'>{_esc(v['checksum'][:10])}</td>"
                         f"<td class='l note'>{_esc(v.get('note',''))}</td></tr>")
            o.append("</tbody></table>")
        for dname, dim in r.dims.items():
            if dim.detail:
                head = "" if dim.ok() else " &mdash; FAILING"
                o.append(f'<ul class="det"><li><b class="mono">{dname}</b>{head}<ul>')
                for det in dim.detail:
                    o.append(f"<li>{_esc(det.splitlines()[0])}</li>")
                o.append("</ul></li></ul>")
        o.append("</div></details>")

    # --- legend ---------------------------------------------------------------------------
    o.append("<h2>What each dimension means</h2><ul class='det legend'>")
    for d in DIMS:
        o.append(f"<li><b>{d}</b> &mdash; {_DIM_HELP[d]}</li>")
    o.append("</ul>")
    o.append('<footer>Generated by <code>python -m kernelbench eval</code>. '
             'The mutation column is the one to read first: a suite where every kernel passes '
             'and no injected bug is caught has established nothing.</footer>')
    o.append("</div>")
    return "\n".join(o)
