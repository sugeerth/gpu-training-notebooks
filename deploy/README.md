# deploy/

Three ways to run `servingkit`, and one checker that keeps them consistent with each other and
with the code.

| | what it is | when |
|---|---|---|
| [`Dockerfile`](Dockerfile) | the API image — Python plus this repo, no other dependencies | always |
| [`Dockerfile.kernels`](Dockerfile.kernels) | the same repo plus `g++`, for verifying kernels | CI, batch, fan-out |
| [`docker-compose.yml`](docker-compose.yml) | API + static console + a one-shot verifier | a laptop |
| [`k8s/`](k8s) | Deployment, Service, HPA, Ingress, NetworkPolicy, PVC, a fan-out Job, a nightly CronJob | a cluster |
| [`verify.py`](verify.py) | ten static checks across all of the above | before you push |

### What is in `k8s/`

| file | objects | why it exists |
|---|---|---|
| `00-namespace.yaml` | Namespace | everything else is namespaced into it |
| `10-config.yaml` | ConfigMap | deployment shape only — see the design note below |
| `20-api.yaml` | Deployment, Service, PodDisruptionBudget, HorizontalPodAutoscaler | the service itself, 2–10 replicas on CPU |
| `30-storage.yaml` | PersistentVolumeClaim | where the scorecard lands; RWX, and optional |
| `40-verify-job.yaml` | Job ×2 | the fan-out, plus the merge that combines its shards |
| `50-drift-cronjob.yaml` | CronJob | nightly: does the package still agree with the notebooks |
| `60-ingress.yaml` | Ingress | nginx-ingress; adjust host and class for your cluster |
| `70-networkpolicy.yaml` | NetworkPolicy | DNS egress and nothing else |
| `kustomization.yaml` | Kustomization | `kubectl apply -k deploy/k8s` — Jobs excluded on purpose |

## Locally

```bash
docker compose -f deploy/docker-compose.yml up --build
open http://localhost:8080                 # the console
curl -s localhost:8000/v1/catalog | jq .   # the API
```

And once, to produce a scorecard the API will then serve at `/v1/scorecard`:

```bash
docker compose -f deploy/docker-compose.yml --profile verify run --rm verifier
```

The verifier is a profile-gated one-shot rather than a service. A batch job dressed as a service
is a lie that surfaces as a crash loop at three in the morning.

## On a cluster

```bash
docker build -f deploy/Dockerfile        -t servingkit/api:0.1.0     .
docker build -f deploy/Dockerfile.kernels -t servingkit/kernels:0.1.0 .

kubectl apply -k deploy/k8s
kubectl -n servingkit rollout status deploy/servingkit-api
```

Then the fan-out — this is the part that spins up one pod per shard of the kernel set:

```bash
kubectl apply -f deploy/k8s/40-verify-job.yaml
kubectl -n servingkit logs -f job/servingkit-verify
kubectl -n servingkit wait --for=condition=complete job/servingkit-verify --timeout=15m
```

Twenty-two CUDA programs, each compiled with `g++` against the CPU shim and checked against its
own double-precision reference. About four minutes in one pod; about forty seconds across six.

`completionMode: Indexed` is what makes this work without a queue: Kubernetes hands each pod a
distinct `JOB_COMPLETION_INDEX`, the pipeline deals kernels round-robin by that index, and no two
pods pick the same work. Round-robin rather than contiguous slicing, because the kernels differ in
cost by an order of magnitude and contiguous slicing would put the four attention kernels in one
pod.

The Jobs are deliberately **excluded from `kustomization.yaml`**. A Job is immutable once created,
so re-applying the directory after a code change would fail on them instead of rolling the
Deployment.

## The data path, end to end

```
   workload spec ──► resolve ──► plan ──► interact ──► verify ──► drift ──► scorecard.json
   {"kind":"agent",     │          │         │           │         │            │
    "turns":30,         │          │         │           │         │            └─► GET /v1/scorecard
    "fanout":4}         │          │         │           │         │                     │
                        │          │         │           │         │                     ▼
              properties│   ladder,│  pairwise│   the kernels│  package│              the console
               derived  │  the money│  gains  │  this plan   │  vs the │
                        │           │         │  names, compiled│ notebooks│
```

One command runs all of it:

```bash
python -m servingkit pipeline --agent --turns 30 --fanout 4 -o scorecard.json
```

```
  resolve    ok        0.06s
  plan       ok        0.00s
  interact   ok        0.00s
  verify     ok       25.11s
  drift      ok        0.04s
```

Stage 4 is why this is a pipeline and not a report. A plan that recommends cascade attention is a
claim about `13_prefix_attention.cu`, so the pipeline compiles that file and checks it before the
claim ships. Stage 3 is a gate, not a warning: an antagonistic lever pair with no declared shared
resource means a lever is lying about what it spends, and a plan built on a lying lever is advice
that costs money.

The scorecard carries a `provenance` block stating whether any timing in it came from real
hardware. It currently always says no.

## Verifying without a daemon

```bash
python deploy/verify.py --verbose
```

Ten check groups, all static:

1. every manifest parses and carries `apiVersion` / `kind` / `metadata.name`
2. every `servingkit/*` image is built by a compose service, at a tag the kustomization pins
3. every `COPY` source exists in the build context
4. every container command is a subcommand the CLI actually has — including inside shell wrappers
5. ports agree across `EXPOSE`, compose, the Deployment, the Service and all three probes
6. every environment variable set is one the code reads, and vice versa
7. probe paths are routes the API actually serves
8. volume mounts resolve to declared volumes; every `claimName` resolves to a PVC in the set
9. the fan-out's `SHARDS` matches `completions`, and `completions > 1` implies `Indexed`
10. every pod runs non-root, drops all capabilities, and has `requests <= limits`

**What it does not do:** build an image or talk to a cluster. The environment this was written in
has a Docker client and no daemon, so **the images here have never been built**. `docker build`
and `kubectl apply --dry-run=server` are the checks for that, and they need a daemon and a cluster
respectively. Treat the manifests as reviewed and cross-consistent, not as proven to run.

## Design notes worth knowing before you change something

**The API image cannot compile kernels.** No `g++`, and `SERVINGKIT_ENABLE_KERNELS=0`. Kernel
verification lives in the other image because it needs a toolchain and the API does not, and
merging them would put a C++ compiler in the container that faces the internet. The
`/v1/kernels/{name}/run` route exists for local and batch use and returns 403 when disabled.

**Readiness runs the models; liveness does not.** `/readyz` evaluates a plan and checks the
levers' declarations against measurement, so a pod whose arithmetic broke is pulled from the
Service rather than serving wrong answers. `/healthz` does no work at all, because a slow liveness
probe restarts healthy pods under load.

**The ConfigMap holds deployment shape, not the model.** No catalogs, no levers, no efficiency
constants. Those live in the package where `servingkit check` covers them; putting them in a
ConfigMap would let a cluster operator silently disagree with the tests.

**The API needs no egress.** Every model in it is arithmetic over tables compiled into the
package, so `70-networkpolicy.yaml` permits DNS and nothing else. A dependency that later reaches
for the network then fails loudly in staging rather than quietly in production.

**`ReadWriteMany` is the one demanding requirement.** The fan-out writes from several pods and the
API replicas read. If your storage class has no RWX mode, delete `30-storage.yaml` and the
`scorecard` volume in `20-api.yaml`; `/v1/scorecard` then returns 404, which is handled behaviour
rather than a failure.

## Routes

| | |
|---|---|
| `GET /healthz` | liveness — no work |
| `GET /readyz` | readiness — the models answer and agree with themselves |
| `GET /v1/catalog` | GPUs, models, precisions, resources, properties |
| `GET /v1/levers` | every lever and its four declarations |
| `POST /v1/workload` | derive properties, bottleneck, raggedness |
| `POST /v1/plan` | the recommended stack, costed |
| `POST /v1/ladder` | one rung per lever, ordered by effort |
| `POST /v1/interactions` | pairwise gains with the predicted conflict |
| `POST /v1/cheapest` | feasible deployments, cheapest per million tokens first |
| `GET /v1/kernels` | what kernels exist and which lever names each |
| `POST /v1/kernels/{name}/run` | compile and run one — 403 when disabled, 429 when busy |
| `GET /v1/scorecard` | the last pipeline run, if one has been written |
| `GET /metrics` | Prometheus text format |

Every POST body is the same workload spec the console and the CLI use, so there is one vocabulary
across the system:

```json
{"kind": "agent", "model": "Llama-3.1-8B", "turns": 30, "fanout": 4,
 "tool_result_tokens": 2000, "replayed": true, "gpu": "H100 SXM"}
```
