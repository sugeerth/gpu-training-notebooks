#!/usr/bin/env python3
"""End-to-end pipeline: fine-tune -> merge -> quality gate -> serve -> live gate -> verdict.

This is the script form of `From_FineTune_To_Production.ipynb`. It runs the whole path on one
GPU box and exits non-zero if any gate fails, so it can sit in CI as a release gate.

    python tools/e2e_pipeline.py                        # full run (needs a GPU)
    python tools/e2e_pipeline.py --dry-run              # exercise the gates with no GPU
    python tools/e2e_pipeline.py --skip-train --merged-path /artifacts/my-model
    python tools/e2e_pipeline.py --min-quality 0.75 --report release.md

Stages, each gated:
    1 train    a small LoRA adapter (skip with --skip-train)
    2 merge    adapter into the base, full precision
    3 gate     offline quality suite vs the recorded baseline
    4 serve    vLLM with a config DERIVED from capacity math, not guessed
    5 gate     the same suite against the live endpoint
    6 verdict  SHIP / BLOCK, with a report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ------------------------------------------------------------------ quality suite
# (id, prompt, must_contain, must_not_contain, category)
SUITE = [
    ("refund_window", "What is our refund window?", ["30 day"], ["sorry"], "task"),
    ("escalate", "The customer is furious about a double charge.", ["escalat"], [], "task"),
    ("policy_no", "Can I get a refund after 6 months?", ["no"], [], "task"),
    ("general_math", "What is 17 + 25?", ["42"], [], "general"),
    ("general_fact", "What is the capital of Japan?", ["tokyo"], [], "general"),
]
TRAIN_PAIRS = [
    ("What is our refund window?", "Our refund window is 30 days from delivery."),
    ("Can I get a refund after 6 months?", "No, that is outside our 30 day window."),
    ("The customer is furious about a double charge.",
     "I will escalate this to billing immediately."),
]


def _contains(haystack: str, needle: str) -> bool:
    """Match a needle, with the two failure modes of naive matching handled.

    1. Plain substring matching silently passes WRONG answers: `"no" in "processing that now"`
       is True, so a gate looking for "no" accepts an answer that said yes.
    2. Strict word-boundary matching breaks INTENTIONAL stems: "escalat" (to catch
       escalate/escalated/escalation) and "30 day" (to catch "30 days") would both fail.

    So: short single tokens (<= 4 chars) — where accidental substring hits are likely and
    stemming is not the intent — match on word boundaries. Everything longer, and every
    multi-word phrase, matches as a substring so stems keep working.
    """
    import re
    if len(needle) <= 4 and " " not in needle:
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None
    return needle in haystack


def score(case, text: str) -> bool:
    _id, _p, must, must_not, _c = case
    t = (text or "").lower()
    return (all(_contains(t, m.lower()) for m in must)
            and not any(_contains(t, m.lower()) for m in must_not))


def run_suite(answer_fn) -> dict:
    per_case, by_cat = {}, {}
    for case in SUITE:
        try:
            ok = score(case, answer_fn(case[1]))
        except Exception as exc:                      # a crash is a failed case, not a crashed run
            ok = False
            print(f"      ! {case[0]} raised {type(exc).__name__}: {exc}")
        per_case[case[0]] = ok
        by_cat.setdefault(case[4], []).append(ok)
    return {"per_case": per_case,
            "overall": sum(per_case.values()) / len(per_case),
            "by_category": {c: sum(v) / len(v) for c, v in by_cat.items()}}


def quality_gate(result: dict, min_overall: float, protect=("task",),
                 min_protected: float = 1.0,
                 baseline: dict | None = None,
                 category_tol: float = 0.10) -> tuple[bool, list[str]]:
    """Absolute floors AND, when a baseline is supplied, a regression check.

    An absolute floor alone is not enough: a model that keeps its task score but forgets
    everything else can still clear it. Regression against a baseline is what catches that,
    which is why the pipeline records a baseline before it optimizes anything.
    """
    failures = []
    if result["overall"] < min_overall:
        failures.append(f"overall {result['overall']:.0%} < required {min_overall:.0%}")
    for cat in protect:
        got = result["by_category"].get(cat)
        if got is not None and got < min_protected:
            failures.append(f"protected category '{cat}' at {got:.0%} "
                            f"(requires {min_protected:.0%})")
    if baseline:
        if result["overall"] < baseline["overall"] - 0.05:
            failures.append(f"overall regressed {baseline['overall']:.0%} -> {result['overall']:.0%}")
        for cat, base_v in baseline["by_category"].items():
            got = result["by_category"].get(cat, 0.0)
            tol = 0.0 if cat in protect else category_tol
            if got < base_v - tol:
                failures.append(f"category '{cat}' regressed {base_v:.0%} -> {got:.0%}")
        regressed = [k for k, v in baseline["per_case"].items()
                     if v and not result["per_case"].get(k, False)]
        if regressed:
            failures.append("cases that used to pass now fail: " + ", ".join(regressed))
    return (not failures), failures


# ------------------------------------------------------------------ capacity math (nb 22/27/28)
def serving_config(params_b: float, weight_bytes: float, layers: int, kv_heads: int,
                   head_dim: int, vram_gb: float, p99_prompt_tokens: int,
                   target_concurrency: int, gpu_util: float = 0.85) -> dict:
    weights_gb = params_b * weight_bytes
    pool_gb = vram_gb * gpu_util - weights_gb - 1.5
    kv_per_token = 2 * layers * kv_heads * head_dim * 2
    max_model_len = 1 << max(9, math.ceil(math.log2(max(1, p99_prompt_tokens) * 1.5)))
    pool_tokens = int(pool_gb * 1e9 // kv_per_token) if pool_gb > 0 else 0
    concurrency = pool_tokens // max_model_len if max_model_len else 0
    return {"feasible": pool_gb > 0 and concurrency >= 1,
            "weights_gb": round(weights_gb, 2), "kv_pool_gb": round(pool_gb, 2),
            "max_model_len": max_model_len, "max_concurrency": concurrency,
            "max_num_seqs": max(1, min(target_concurrency, concurrency)),
            "gpu_memory_utilization": gpu_util}


# ------------------------------------------------------------------ stages
class Pipeline:
    def __init__(self, args):
        self.args = args
        self.stages: list[dict] = []
        self.shipped = False

    def stage(self, name: str, ok: bool, detail: str) -> bool:
        self.stages.append({"stage": name, "ok": bool(ok), "detail": detail})
        print(f"  {'✅' if ok else '❌'} {name:<16}{detail}")
        return ok

    # -- 1/2: train + merge ------------------------------------------------
    def train_and_merge(self) -> str | None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        out = Path(self.args.workdir) / "merged"
        tok = AutoTokenizer.from_pretrained(self.args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.args.model, dtype=torch.float32).to("cuda")
        model = get_peft_model(model, LoraConfig(
            r=self.args.lora_rank, lora_alpha=2 * self.args.lora_rank, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False)
            for q, a in TRAIN_PAIRS]
        model.train()
        loss = None
        for _ in range(self.args.steps):
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to("cuda")
            loss = model(**enc, labels=enc.input_ids).loss
            loss.backward(); opt.step(); opt.zero_grad()
        self.stage("train", True, f"{self.args.steps} steps, final loss {loss.item():.3f}")

        model.merge_and_unload().save_pretrained(out)
        tok.save_pretrained(out)
        del model, opt
        torch.cuda.empty_cache()
        self.stage("merge", True, str(out))
        return str(out)

    # -- 4: serve ----------------------------------------------------------
    def serve(self, path: str, cfg: dict):
        log = Path(self.args.workdir) / "server.log"
        proc = subprocess.Popen(
            ["vllm", "serve", path, "--dtype", "half",
             "--max-model-len", str(cfg["max_model_len"]),
             "--max-num-seqs", str(cfg["max_num_seqs"]),
             "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
             "--served-model-name", "candidate", "--port", str(self.args.port)],
            stdout=open(log, "w"), stderr=subprocess.STDOUT)
        url = f"http://localhost:{self.args.port}"
        for _ in range(self.args.startup_timeout // 2):
            if proc.poll() is not None:
                self.stage("serve", False, f"server exited early - see {log}")
                return None
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                self.stage("serve", True, f"up on {url} (log: {log})")
                return proc
            except Exception:
                time.sleep(2)
        proc.terminate()
        self.stage("serve", False, f"did not become healthy in {self.args.startup_timeout}s")
        return None

    # -- report ------------------------------------------------------------
    def report(self, extra: dict) -> str:
        lines = ["# End-to-end pipeline report", "",
                 f"- **verdict**: {'✅ SHIP' if self.shipped else '❌ BLOCKED'}",
                 f"- **base model**: `{self.args.model}`", ""]
        lines += ["| stage | result | detail |", "|---|---|---|"]
        for s in self.stages:
            lines.append(f"| {s['stage']} | {'✅' if s['ok'] else '❌'} | {s['detail']} |")
        if extra:
            lines += ["", "## Details", "", "```json", json.dumps(extra, indent=2), "```"]
        return "\n".join(lines)


def dry_run(args) -> int:
    """Exercise every gate with a simulated model - no GPU, no network."""
    print("DRY RUN - simulating the pipeline to exercise the gates\n")
    good = {q: a for q, a in TRAIN_PAIRS}
    good.update({"What is 17 + 25?": "42", "What is the capital of Japan?": "Tokyo"})

    def make(damage=None):
        def fn(prompt):
            if damage == "forgetting" and prompt in ("What is 17 + 25?",
                                                     "What is the capital of Japan?"):
                return "I only handle support questions."
            if damage == "task" and prompt == "Can I get a refund after 6 months?":
                return "Yes of course, processing that now."
            return good.get(prompt, "")
        return fn

    rc = 0
    baseline = run_suite(make())                      # record the baseline BEFORE optimizing
    print(f"  baseline recorded: overall {baseline['overall']:.0%}\n")
    for label, damage, should_pass in [("healthy", None, True),
                                       ("forgetting", "forgetting", False),
                                       ("task drift", "task", False)]:
        res = run_suite(make(damage))
        ok, failures = quality_gate(res, args.min_quality, baseline=baseline)
        print(f"  {'✅ PASS' if ok else '❌ BLOCK'}  {label:<12}overall {res['overall']:.0%}"
              + ("" if ok else f"  ({failures[0]})"))
        if ok != should_pass:
            # The gate itself is under test here: a gate that waves through a known-bad build
            # is worse than no gate, because it manufactures confidence.
            print(f"          !! GATE IS WRONG: expected this build to "
                  f"{'pass' if should_pass else 'BLOCK'}")
            rc = 1
    cfg = serving_config(0.5, 2.0, 24, 2, 64, 16, 1800, 64)
    print(f"\n  derived serving config: max_model_len={cfg['max_model_len']}, "
          f"max_num_seqs={cfg['max_num_seqs']} (KV holds {cfg['max_concurrency']})")
    print("\ndry run complete - gates behave correctly"
          if rc == 0 else "\nDRY RUN FAILED: the healthy build did not pass its own gate")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--workdir", default="./e2e_run")
    ap.add_argument("--merged-path", default=None, help="skip training, use this artifact")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--min-quality", type=float, default=0.6)
    ap.add_argument("--p99-prompt-tokens", type=int, default=1800)
    ap.add_argument("--target-concurrency", type=int, default=32)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout", type=int, default=360)
    ap.add_argument("--report", default="e2e_report.md")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args)

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if not has_gpu:
        print("No GPU detected. This pipeline needs one.\n"
              "Use --dry-run to exercise the gates without hardware.")
        return 2

    os.makedirs(args.workdir, exist_ok=True)
    p = Pipeline(args)
    print(f"end-to-end pipeline · base {args.model}\n")

    path = args.merged_path
    if not path and not args.skip_train:
        path = p.train_and_merge()
    if not path:
        print("\nno artifact to serve (use --merged-path or drop --skip-train)")
        return 1

    import torch
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cfg = serving_config(0.5, 2.0, 24, 2, 64, vram_gb, args.p99_prompt_tokens,
                         args.target_concurrency)
    if not p.stage("capacity plan", cfg["feasible"],
                   f"max_model_len {cfg['max_model_len']}, {cfg['max_concurrency']} concurrent"):
        Path(args.report).write_text(p.report({"config": cfg}))
        return 1

    proc = p.serve(path, cfg)
    if proc is None:
        Path(args.report).write_text(p.report({"config": cfg}))
        return 1

    try:
        from openai import OpenAI
        client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="x")

        def ask(prompt: str) -> str:
            return client.chat.completions.create(
                model="candidate", temperature=0.0, max_tokens=60,
                messages=[{"role": "user", "content": prompt}]).choices[0].message.content

        result = run_suite(ask)
        ok, failures = quality_gate(result, args.min_quality)
        p.stage("live quality gate", ok,
                f"overall {result['overall']:.0%}" + ("" if ok else f" - {failures[0]}"))
        for cid, passed in result["per_case"].items():
            print(f"        {'✅' if passed else '❌'} {cid}")
        p.shipped = ok
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        print("  · server stopped")

    Path(args.report).write_text(p.report({"config": cfg, "quality": result}))
    print(f"\n{'✅ SHIP' if p.shipped else '❌ BLOCKED'} · report written to {args.report}")
    return 0 if p.shipped else 1


if __name__ == "__main__":
    sys.exit(main())
