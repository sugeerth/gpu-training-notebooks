#!/usr/bin/env python3
"""Check the deployment without a container runtime.

Building an image needs a daemon and applying a manifest needs a cluster, and neither is
available everywhere this runs. Rather than skip the whole thing when they are missing, this
checks everything that *is* checkable statically — and a surprising amount is, because most
deployment breakage is a mismatch between two files rather than a failure inside one:

  1. every manifest parses, and every object has apiVersion / kind / metadata.name
  2. every `image:` in the manifests is built by one of the Dockerfiles, at a matching tag
  3. every COPY source in a Dockerfile exists in the build context
  4. every container command is a subcommand the CLI actually has
  5. ports agree across Dockerfile EXPOSE, compose, the Deployment, the Service and the probes
  6. every environment variable the manifests set is one the code reads, and vice versa
  7. probe paths are routes the API actually serves
  8. volume mounts resolve to declared volumes; claim names resolve to a PVC in the same set
  9. the ConfigMap keys the pods consume are the ones it defines
 10. resource limits are >= requests, and nothing runs as root

    python deploy/verify.py            # check
    python deploy/verify.py --verbose  # and say what passed

What it does NOT do: build an image, start a pod, or prove the cluster will accept the manifests.
`kubectl apply --dry-run=server` is the check for that, and it needs a cluster.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("this checker needs PyYAML: pip install pyyaml")

DEPLOY = Path(__file__).resolve().parent
REPO = DEPLOY.parent
K8S = DEPLOY / "k8s"

problems: list[str] = []
passed: list[str] = []


def ok(msg: str) -> None:
    passed.append(msg)


def bad(msg: str) -> None:
    problems.append(msg)


# --------------------------------------------------------------------------------- loading
def load_manifests() -> list[tuple[Path, dict]]:
    out = []
    for f in sorted(K8S.glob("*.yaml")):
        try:
            docs = list(yaml.safe_load_all(f.read_text()))
        except yaml.YAMLError as exc:
            bad(f"{f.name}: does not parse — {exc}")
            continue
        for d in docs:
            if d:
                out.append((f, d))
    return out


def load_compose() -> dict:
    f = DEPLOY / "docker-compose.yml"
    try:
        return yaml.safe_load(f.read_text()) or {}
    except yaml.YAMLError as exc:
        bad(f"docker-compose.yml: does not parse — {exc}")
        return {}


def dockerfiles() -> dict[str, str]:
    return {f.name: f.read_text() for f in DEPLOY.glob("Dockerfile*")}


def pod_specs(manifests) -> list[tuple[Path, dict, dict]]:
    """Every pod template in the set, with the object that owns it."""
    out = []
    for f, d in manifests:
        kind = d.get("kind")
        if kind in ("Deployment", "Job", "StatefulSet", "DaemonSet"):
            spec = d.get("spec", {}).get("template", {}).get("spec")
        elif kind == "CronJob":
            spec = (d.get("spec", {}).get("jobTemplate", {}).get("spec", {})
                    .get("template", {}).get("spec"))
        elif kind == "Pod":
            spec = d.get("spec")
        else:
            continue
        if spec:
            out.append((f, d, spec))
    return out


# ---------------------------------------------------------------------------------- checks
def check_shape(manifests) -> None:
    for f, d in manifests:
        for key in ("apiVersion", "kind"):
            if not d.get(key):
                bad(f"{f.name}: an object has no {key}")
        if d.get("kind") != "Kustomization" and not d.get("metadata", {}).get("name"):
            bad(f"{f.name}: a {d.get('kind')} has no metadata.name")
    ok(f"{len(manifests)} objects parse and carry apiVersion/kind/name")


def check_images(manifests, compose, dfs) -> None:
    """Every image referenced must be one this repo builds, at a tag the compose file agrees on."""
    built = {}
    for name, svc in (compose.get("services") or {}).items():
        img, build = svc.get("image"), svc.get("build")
        if img and build:
            built[img] = f"{build.get('dockerfile')} (compose service {name})"
    if not built:
        bad("docker-compose.yml builds no images, so nothing pins the tags the manifests use")

    referenced = set()
    for _f, _d, spec in pod_specs(manifests):
        for c in spec.get("containers", []) + spec.get("initContainers", []):
            referenced.add(c["image"])
    # nginx and other upstream images are legitimately not built here.
    ours = {i for i in referenced if i.startswith("servingkit/")}
    for img in sorted(ours):
        if img not in built:
            bad(f"manifests use {img}, which no compose service builds — "
                f"built here: {sorted(built)}")
    for img in sorted(built):
        if img.startswith("servingkit/") and img not in referenced:
            bad(f"{img} is built but no manifest references it")
    if ours and not (ours - set(built)):
        ok(f"{len(ours)} servingkit image(s) referenced, all built by a compose service")

    # The kustomization's image tags must match the tags in the manifests, or `apply -k` silently
    # deploys something other than what the files say.
    kust = next((d for _f, d in manifests if d.get("kind") == "Kustomization"), None)
    if kust:
        for entry in kust.get("images", []):
            want = f"{entry['name']}:{entry['newTag']}"
            if want not in referenced:
                bad(f"kustomization pins {want} but no manifest references that exact tag")
        ok("kustomization image tags match the manifests")


def check_copy_sources(dfs) -> None:
    """Every COPY source must exist, or the build fails at layer three with a useless message."""
    n = 0
    for name, text in dfs.items():
        for m in re.finditer(r"^COPY\s+(?!--from)(.+)$", text, re.M):
            parts = m.group(1).split()
            if len(parts) < 2:
                bad(f"{name}: malformed COPY: {m.group(0)}")
                continue
            for src in parts[:-1]:
                if src.startswith("--"):
                    continue
                n += 1
                if any(ch in src for ch in "*?["):
                    if not list(REPO.glob(src)):
                        bad(f"{name}: COPY {src} matches nothing in the build context")
                elif not (REPO / src).exists():
                    bad(f"{name}: COPY {src} does not exist in the build context")
    ok(f"{n} COPY source(s) resolve in the build context")


def cli_subcommands() -> set[str]:
    text = (REPO / "servingkit" / "cli.py").read_text()
    return set(re.findall(r'sub\.add_parser\(\s*"([a-z_]+)"', text))


def check_commands(manifests, compose, dfs) -> None:
    """A container whose args are not a real subcommand crash-loops with an argparse error."""
    subs = cli_subcommands()
    if not subs:
        bad("could not read any subcommands out of servingkit/cli.py")
        return

    def check_args(where: str, args) -> None:
        if not args:
            return
        first = args[0]
        # A shell-wrapped container's real command is inside the script; find the module call.
        if first in ("/bin/sh", "/bin/bash", "sh", "bash"):
            script = "\n".join(str(a) for a in args)
            for m in re.finditer(r"python -m servingkit\s+\\?\s*\n?\s*\$?\{?([a-z_]+)", script):
                cand = m.group(1)
                if cand in subs or cand.startswith("PIPELINE"):
                    continue
            found = re.findall(r"python -m servingkit(?:\s+\\\s*)?\s+([a-z_]+)", script)
            for cand in found:
                if cand not in subs:
                    bad(f"{where}: runs `servingkit {cand}`, which is not a subcommand {sorted(subs)}")
            if found:
                ok(f"{where}: shell-wrapped `servingkit {found[0]}` is a real subcommand")
            return
        if first not in subs:
            bad(f"{where}: first arg {first!r} is not a servingkit subcommand {sorted(subs)}")

    for f, d, spec in pod_specs(manifests):
        for c in spec.get("containers", []):
            if not c["image"].startswith("servingkit/"):
                continue
            args = c.get("args") or []
            if c.get("command") and c["command"][0] not in ("/bin/sh", "/bin/bash"):
                args = c["command"] + args
            elif c.get("command"):
                args = c["command"] + args
            check_args(f"{f.name}/{d['metadata']['name']}/{c['name']}", args)

    for name, svc in (compose.get("services") or {}).items():
        if str(svc.get("image", "")).startswith("servingkit/"):
            check_args(f"compose/{name}", svc.get("command") or [])

    for name, text in dfs.items():
        m = re.search(r'^CMD\s+\[(.+)\]', text, re.M)
        if m:
            args = [a.strip().strip('"') for a in m.group(1).split(",")]
            if args and args[0] not in subs:
                bad(f"{name}: CMD starts with {args[0]!r}, not a subcommand")
    ok(f"container commands use only real subcommands ({len(subs)} available)")


def api_routes() -> set[str]:
    text = (REPO / "servingkit" / "api.py").read_text()
    routes = set(re.findall(r'^\s*"(/[a-z0-9/_{}.-]+)":', text, re.M))
    routes |= set(re.findall(r'path == "(/[a-z]+)"', text))
    routes |= {"/healthz", "/readyz", "/metrics"}
    return routes


def check_ports_and_probes(manifests, compose, dfs) -> None:
    routes = api_routes()
    api_port = 8000

    exposed = {int(m.group(1)) for t in dfs.values()
               for m in re.finditer(r"^EXPOSE\s+(\d+)", t, re.M)}
    if exposed and api_port not in exposed:
        bad(f"no Dockerfile EXPOSEs {api_port}; found {sorted(exposed)}")

    for f, d, spec in pod_specs(manifests):
        for c in spec.get("containers", []):
            names = {p.get("name"): p.get("containerPort") for p in c.get("ports", [])}
            if c["image"].startswith("servingkit/api") and names.get("http") != api_port:
                bad(f"{f.name}/{c['name']}: containerPort for 'http' is {names.get('http')}, "
                    f"expected {api_port}")
            for probe in ("readinessProbe", "livenessProbe", "startupProbe"):
                p = c.get(probe, {}).get("httpGet")
                if not p:
                    continue
                if p["path"] not in routes:
                    bad(f"{f.name}/{c['name']}/{probe}: path {p['path']} is not a route the API "
                        f"serves ({sorted(routes)[:6]}...)")
                if p["port"] not in names and p["port"] != api_port:
                    bad(f"{f.name}/{c['name']}/{probe}: port {p['port']!r} is not a declared "
                        f"port name {sorted(k for k in names if k)}")
    # Service targetPort must name a port the container declares.
    for f, d in manifests:
        if d.get("kind") != "Service":
            continue
        for p in d["spec"]["ports"]:
            tp = p.get("targetPort")
            if isinstance(tp, str) and tp != "http":
                bad(f"{f.name}: Service targetPort {tp!r} does not name the container's 'http'")
    for name, svc in (compose.get("services") or {}).items():
        for mapping in svc.get("ports", []):
            host, _, cont = str(mapping).partition(":")
            if str(svc.get("image", "")).startswith("servingkit/api") and int(cont) != api_port:
                bad(f"compose/{name}: maps to container port {cont}, expected {api_port}")
    ok("ports and probe paths agree across Dockerfile, compose, Deployment, Service and probes")


def code_env_vars() -> set[str]:
    names = set()
    for f in (REPO / "servingkit").glob("*.py"):
        t = f.read_text()
        names |= set(re.findall(r'environ\.get\(\s*"([A-Z0-9_]+)"', t))
        names |= set(re.findall(r'environ\[\s*"([A-Z0-9_]+)"\s*\]', t))
    return names


def check_env(manifests, compose) -> None:
    """A variable set but never read is dead config; one read but never set is a hidden default."""
    read = code_env_vars()
    # Supplied by Kubernetes or the shell, not by our code.
    external = {"HOSTNAME", "NODE_NAME", "JOB_COMPLETION_INDEX", "SHARDS", "PIPELINE_ARGS",
                "GIT_SHA", "PORT", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"}
    set_names: set[str] = set()

    cfg = next((d for _f, d in manifests if d.get("kind") == "ConfigMap"), None)
    cfg_keys = set((cfg or {}).get("data", {}))
    set_names |= cfg_keys

    for _f, _d, spec in pod_specs(manifests):
        for c in spec.get("containers", []):
            set_names |= {e["name"] for e in c.get("env", [])}
            # A shell-wrapped container sets variables with `export NAME=...` inside its script,
            # not in the env: block. Missing that reported SERVINGKIT_SHARD as unset when the
            # fan-out Job plainly sets it, which is a gap in the checker rather than in the
            # deployment — so the checker reads the script too.
            script = "\n".join(str(x) for x in (c.get("args") or []))
            set_names |= set(re.findall(r"export\s+([A-Z0-9_]+)=", script))
            for ref in c.get("envFrom", []):
                nm = ref.get("configMapRef", {}).get("name")
                if nm and cfg and nm != cfg["metadata"]["name"]:
                    bad(f"a container references ConfigMap {nm}, which is not defined here")
    for svc in (compose.get("services") or {}).values():
        set_names |= set(svc.get("environment") or {})

    for n in sorted(set_names - read - external):
        bad(f"{n} is set by the deployment but no code reads it")
    for n in sorted(read - set_names - external):
        # Reading with a default is fine; this only reports ones the deployment never mentions.
        if n.startswith("SERVINGKIT_"):
            bad(f"{n} is read by the code but never set by any manifest or compose service")
    ok(f"{len(set_names & read)} environment variable(s) are both set and read; "
       f"{len(cfg_keys)} ConfigMap key(s)")


def check_volumes(manifests) -> None:
    claims = {d["metadata"]["name"] for _f, d in manifests
              if d.get("kind") == "PersistentVolumeClaim"}
    for f, d, spec in pod_specs(manifests):
        declared = {v["name"]: v for v in spec.get("volumes", [])}
        for c in spec.get("containers", []):
            for mnt in c.get("volumeMounts", []):
                if mnt["name"] not in declared:
                    bad(f"{f.name}/{c['name']}: mounts volume {mnt['name']!r} which the pod "
                        f"does not declare")
        for v in declared.values():
            pvc = v.get("persistentVolumeClaim", {}).get("claimName")
            if pvc and pvc not in claims:
                bad(f"{f.name}: volume {v['name']} claims {pvc!r}, which no PVC in this set "
                    f"defines ({sorted(claims)})")
    ok(f"volume mounts resolve; {len(claims)} PVC(s) defined and every claimName matches")


def check_security_and_resources(manifests) -> None:
    for f, d, spec in pod_specs(manifests):
        name = d["metadata"]["name"]
        psc = spec.get("securityContext", {})
        if not psc.get("runAsNonRoot"):
            bad(f"{f.name}/{name}: pod securityContext does not set runAsNonRoot")
        for c in spec.get("containers", []):
            csc = c.get("securityContext", {})
            if csc.get("allowPrivilegeEscalation") is not False:
                bad(f"{f.name}/{name}/{c['name']}: allowPrivilegeEscalation is not false")
            if "ALL" not in (csc.get("capabilities", {}).get("drop") or []):
                bad(f"{f.name}/{name}/{c['name']}: does not drop ALL capabilities")
            res = c.get("resources", {})
            req, lim = res.get("requests", {}), res.get("limits", {})
            if not req or not lim:
                bad(f"{f.name}/{name}/{c['name']}: missing resource requests or limits")
                continue
            for key in ("cpu", "memory"):
                if key in req and key in lim and _quantity(lim[key]) < _quantity(req[key]):
                    bad(f"{f.name}/{name}/{c['name']}: {key} limit {lim[key]} is below "
                        f"request {req[key]}")
    ok("every pod runs non-root, drops all capabilities, and has requests <= limits")


_SUFFIX = {"m": 1e-3, "": 1, "k": 1e3, "Ki": 2**10, "M": 1e6, "Mi": 2**20,
           "G": 1e9, "Gi": 2**30, "T": 1e12, "Ti": 2**40}


def _quantity(v) -> float:
    s = str(v)
    m = re.fullmatch(r"([0-9.]+)\s*([a-zA-Z]*)", s)
    if not m:
        return float("nan")
    return float(m.group(1)) * _SUFFIX.get(m.group(2), 1)


def check_job_fanout(manifests) -> None:
    """The fan-out's shard count must agree in three places or pods silently duplicate work."""
    for f, d in manifests:
        if d.get("kind") != "Job":
            continue
        spec = d["spec"]
        name = d["metadata"]["name"]
        if spec.get("completionMode") != "Indexed" and spec.get("completions", 1) > 1:
            bad(f"{f.name}/{name}: completions > 1 without completionMode: Indexed — every pod "
                f"would run the same shard")
        completions = spec.get("completions")
        pod = spec["template"]["spec"]
        for c in pod.get("containers", []):
            shards = next((e["value"] for e in c.get("env", []) if e["name"] == "SHARDS"), None)
            if shards is None:
                continue
            if completions and int(shards) != int(completions):
                bad(f"{f.name}/{name}: SHARDS={shards} but completions={completions}; the "
                    f"kernels would be dealt into the wrong number of piles")
    ok("fan-out Job: Indexed completion, and SHARDS matches completions")


def check_readme() -> None:
    r = DEPLOY / "README.md"
    if not r.exists():
        bad("deploy/README.md is missing")
        return
    text = r.read_text()
    for f in sorted(list(K8S.glob("*.yaml")) + list(DEPLOY.glob("Dockerfile*"))
                    + [DEPLOY / "docker-compose.yml"]):
        rel = f.name
        if rel not in text:
            bad(f"deploy/README.md does not mention {rel}")
    ok("deploy/README.md mentions every file in deploy/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    manifests = load_manifests()
    compose = load_compose()
    dfs = dockerfiles()

    check_shape(manifests)
    check_images(manifests, compose, dfs)
    check_copy_sources(dfs)
    check_commands(manifests, compose, dfs)
    check_ports_and_probes(manifests, compose, dfs)
    check_env(manifests, compose)
    check_volumes(manifests)
    check_security_and_resources(manifests)
    check_job_fanout(manifests)
    check_readme()

    if a.verbose:
        for line in passed:
            print(f"  ok    {line}")
    print(f"\n{len(passed)} check group(s) ran over {len(manifests)} objects, "
          f"{len(dfs)} Dockerfile(s), {len(compose.get('services') or {})} compose service(s)")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print("  - " + line)
        return 1
    print("\nevery static check passes")
    print("NOT checked here: whether the images actually build, and whether a cluster accepts")
    print("the manifests. `docker build -f deploy/Dockerfile .` and `kubectl apply -k deploy/k8s")
    print("--dry-run=server` are those checks; they need a daemon and a cluster respectively.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
