---
name: llm-d-preflight-checks
description: Run preflight checks for LLM-D to ensure the environment is properly set up before vLLM is started.
---

Modify llm-d Helm deployment to run llm-d-preflight-checks/scripts/llm-d-preflight-checks.py just before starting vLLM. This script will perform various "preflight" checks to ensure that the environment is properly configured for LLM-D to run vLLM.

The script behavior is controlled by the `LLMD_PREFLIGHT_CHECKS` environment variable:

| Value | Behavior |
|-------|----------|
| unset / `disable` / `none` | Print system diagnostics (env, GPU, CPU, PCI) and exit 0 |
| `pause` | Print diagnostics, then start HTTP server blocking until `/exit` is called |
| `topology` | Print diagnostics and exit (reserved for future topology validation) |
| `nixl` | Print diagnostics and exit (reserved for future NixL checks) |

When in `pause` mode, the HTTP server satisfies K8s health probes and provides:
- `GET /health` — 200 OK (for probes)
- `GET /info` — system diagnostics
- `GET /exit` — shut down server and continue to vLLM startup

The script must be mounted into llm-d pods. To do this, create a ConfigMap with the script and mount it as a volume in the deployment. Then, modify the entrypoint of the container to run the preflight check script before starting vLLM.

## Running preflight checks with llm-d quickstart

Follow the [llm-d quickstart guide](https://llm-d.ai/docs/getting-started/quickstart) to deploy llm-d, then patch the model server deployment to run the preflight checks script before vLLM starts.

### Prerequisites

- llm-d deployed via the quickstart guide (Helm chart + model server kustomization)
- A clone of `llm-d-pd-utils` containing the preflight checks script at `scripts/llm-d-preflight-checks.py` (or the skill directory at `.claude/skills/llm-d-preflight-checks/scripts/llm-d-preflight-checks.py`)

### Step 1: Deploy llm-d via quickstart

```bash
cd /path/to/llm-d
export GAIE_VERSION=v1.5.0
export GUIDE_NAME="quickstart"
export NAMESPACE=<your-namespace>

# Install CRDs
kubectl apply -k "https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=${GAIE_VERSION}"
kubectl create namespace ${NAMESPACE}

# Deploy router
helm install ${GUIDE_NAME} \
    oci://registry.k8s.io/gateway-api-inference-extension/charts/standalone \
    -f guides/recipes/scheduler/base.values.yaml \
    -f guides/optimized-baseline/scheduler/optimized-baseline.values.yaml \
    -n ${NAMESPACE} --version ${GAIE_VERSION}

# Deploy model server
kubectl apply -n ${NAMESPACE} -k guides/optimized-baseline/modelserver/gpu/vllm/base/
```

### Step 2: Create a ConfigMap with the preflight checks script

```bash
kubectl create configmap llm-d-preflight-checks \
  --from-file=llm-d-preflight-checks.py=/path/to/llm-d-pd-utils/.claude/skills/llm-d-preflight-checks/scripts/llm-d-preflight-checks.py \
  -n ${NAMESPACE}
```

### Step 3: Patch the model server deployment

Apply a JSON patch to the deployment that:
1. Adds a volume from the ConfigMap
2. Mounts the script at `/preflight/llm-d-preflight-checks.py`
3. Sets `LLMD_PREFLIGHT_CHECKS=pause` (or another mode)
4. Changes the entrypoint to run the preflight script before vLLM via `&&`

```bash
kubectl patch deployment optimized-baseline-nvidia-gpu-vllm-decode \
  -n ${NAMESPACE} --type=json -p '[
  {
    "op": "add",
    "path": "/spec/template/spec/volumes/-",
    "value": {
      "name": "preflight-checks",
      "configMap": {
        "name": "llm-d-preflight-checks",
        "defaultMode": 493
      }
    }
  },
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/volumeMounts/-",
    "value": {
      "name": "preflight-checks",
      "mountPath": "/preflight"
    }
  },
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/env",
    "value": [
      {
        "name": "LLMD_PREFLIGHT_CHECKS",
        "value": "pause"
      }
    ]
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/command",
    "value": ["bash", "-c"]
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/args",
    "value": [
      "python3 /preflight/llm-d-preflight-checks.py && vllm serve Qwen/Qwen3-32B --disable-access-log-for-endpoints=/health,/metrics,/v1/models --tensor-parallel-size=2 --gpu-memory-utilization=0.95"
    ]
  }
]'
```

This triggers a rolling update. New pods will start the preflight script, which prints system diagnostics (environment variables, GPU topology via `nvidia-smi topo -m`, NVLink status, CPU info via `lscpu`) and then — in `pause` mode — starts an HTTP server on port 8000 that satisfies K8s health probes (`/health` returns 200) while blocking vLLM startup.

The `&&` between the preflight script and `vllm serve` ensures vLLM only starts if the preflight script exits with code 0.

### Step 4: Verify the preflight checks are running

Wait for the rolling update to complete and check pod logs:

```bash
# Check pods are Running and Ready (preflight HTTP server satisfies probes)
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/model=Qwen3-32B

# Verify the preflight script started
kubectl logs -n ${NAMESPACE} <pod-name> | grep "llm-d-preflight-checks.py starting"

# Verify pause mode is active
kubectl logs -n ${NAMESPACE} <pod-name> | grep "LLMD_PREFLIGHT_CHECKS"

# Test the preflight HTTP endpoints
kubectl exec -n ${NAMESPACE} <pod-name> -- curl -s http://localhost:8000/health
# Returns: {"status":"ok"}

kubectl exec -n ${NAMESPACE} <pod-name> -- curl -s http://localhost:8000/info
# Returns: full system diagnostics (env, GPU topology, NVLink, lscpu)
```

### Step 5: Resume vLLM startup

When ready to let vLLM start (after inspecting diagnostics or running network tests), call `/exit` on each pod:

```bash
# Resume all pods matching the label
for pod in $(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/model=Qwen3-32B -o name); do
  kubectl exec -n ${NAMESPACE} $pod -- curl -s http://localhost:8000/exit
done
```

After `/exit` is called, the preflight server shuts down, the script exits with code 0, and `vllm serve` starts normally.

### How it works

The patch modifies the pod spec as follows:

| Original | Patched |
|----------|---------|
| `command: ["vllm", "serve"]` | `command: ["bash", "-c"]` |
| `args: ["Qwen/Qwen3-32B", ...]` | `args: ["python3 /preflight/llm-d-preflight-checks.py && vllm serve Qwen/Qwen3-32B ..."]` |
| No ConfigMap volume | ConfigMap `llm-d-preflight-checks` mounted at `/preflight` (mode 0755) |
| No env vars | `LLMD_PREFLIGHT_CHECKS=pause` |

The ConfigMap volume (`defaultMode: 493` = octal `0755`) ensures the script is executable. The `bash -c` wrapper allows the `&&` chain to work: preflight runs first, and only if it exits 0 does vLLM start.

In `pause` mode, the preflight HTTP server binds to port 8000 (the same port vLLM would use), so K8s startup/liveness/readiness probes pass while vLLM is not yet running. Once `/exit` is called, the server releases port 8000 and vLLM binds to it normally.

### Changing preflight mode without redeployment

To switch from `pause` to a non-blocking mode (e.g., diagnostics-only), patch just the env var:

```bash
kubectl set env deployment/optimized-baseline-nvidia-gpu-vllm-decode \
  -n ${NAMESPACE} LLMD_PREFLIGHT_CHECKS=none
```

This triggers a new rollout where the preflight script prints diagnostics and exits immediately, allowing vLLM to start without manual intervention.

### Cleanup

To remove preflight checks entirely, redeploy the original model server kustomization:

```bash
kubectl apply -n ${NAMESPACE} -k /path/to/llm-d/guides/optimized-baseline/modelserver/gpu/vllm/base/
kubectl delete configmap llm-d-preflight-checks -n ${NAMESPACE}
```

## Running preflight checks with llm-d-benchmark

The [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark) framework can run the preflight checks script automatically before vLLM starts. The framework's step 04 (`04_ensure_model_namespace_prepared.py`) reads all files from `setup/preprocess/` into a ConfigMap named `llm-d-benchmark-preprocesses`, which gets mounted at `/setup/preprocess/` inside pods.

### Step 1: Symlink the script into llm-d-benchmark

Create a symlink from the llm-d-benchmark `setup/preprocess/` directory to the preflight checks script:

```bash
ln -s /path/to/blog7/skills/llm-d-preflight-checks/scripts/llm-d-preflight-checks.py \
      /path/to/llm-d-benchmark/setup/preprocess/llm-d-preflight-checks.py
```

This ensures the script is included in the `llm-d-benchmark-preprocesses` ConfigMap when step 04 runs.

### Step 2: Update the scenario PREPROCESS variable

In your scenario file (e.g., `scenarios/guides/pd-disaggregation2.sh`), set the `LLMDBENCH_VLLM_COMMON_PREPROCESS` variable to include the preflight checks script:

```bash
export LLMDBENCH_VLLM_COMMON_PREPROCESS="python3 /setup/preprocess/set_llmdbench_environment.py; source \$HOME/llmdbench_env.sh; python3 /setup/preprocess/llm-d-preflight-checks.py"
```

The script runs after `set_llmdbench_environment.py` and `source llmdbench_env.sh` so that environment variables like `VLLM_INFERENCE_PORT` are available.

### Step 3: Set LLMD_PREFLIGHT_CHECKS in the scenario

To control the preflight behavior, add `LLMD_PREFLIGHT_CHECKS` to the pod environment variables in the scenario's `LLMDBENCH_VLLM_COMMON_ENVVARS_TO_YAML` block. In `pd-disaggregation2.sh`, the env vars are defined like this:

```bash
export LLMDBENCH_VLLM_COMMON_ENVVARS_TO_YAML=$(mktemp)
cat << EOF > $LLMDBENCH_VLLM_COMMON_ENVVARS_TO_YAML
- name: NCCL_EXCLUDE_IB_HCA
  value: "mlx5_0,mlx5_2,mlx5_4,mlx5_8,mlx5_7,mlx5_10,mlx5_12,mlx5_14,mlx5_16"
- name: NVSHMEM_DEBUG
  value: "INFO"
- name: LLMD_PREFLIGHT_CHECKS
  value: "pause"
EOF
```

Add the `LLMD_PREFLIGHT_CHECKS` entry at the end of the YAML list. Valid values:

| Value | Effect |
|-------|--------|
| `pause` | Print diagnostics, then block with HTTP server until `/exit` is called |
| `disable` or `none` | Print diagnostics and exit immediately |
| `topology` | Print diagnostics and exit (reserved for future topology checks) |
| `nixl` | Print diagnostics and exit (reserved for future NixL checks) |

If `LLMD_PREFLIGHT_CHECKS` is not set at all, the script defaults to printing diagnostics and exiting immediately (no blocking).

When `pause` is set, pods will show as Ready (the preflight HTTP server responds to K8s health probes) but vLLM will **not** start until you explicitly call `/exit` on each pod:

```bash
# Resume a specific pod
kubectl exec -n <namespace> <pod-name> -c vllm -- curl -s http://localhost:8000/exit

# Resume all decode pods
for pod in $(kubectl get pods -n <namespace> -l llm-d.ai/role=decode -o name); do
  kubectl exec -n <namespace> $pod -c vllm -- curl -s http://localhost:8000/exit
done

# Resume all prefill pods
for pod in $(kubectl get pods -n <namespace> -l llm-d.ai/role=prefill -o name); do
  kubectl exec -n <namespace> $pod -c vllm -- curl -s http://localhost:8000/exit
done
```

To disable pause mode and allow normal startup, either remove the `LLMD_PREFLIGHT_CHECKS` entry or change its value to `none`.

### Step 4: Run the standup

```bash
source venv/bin/activate
export LLMDBENCH_HF_TOKEN=<your-token>
export LLMDBENCH_DEPLOY_MODEL_LIST="facebook/opt-125m"

# Teardown any previous deployment
./setup/teardown.sh -c ${PWD}/scenarios/guides/pd-disaggregation2.sh

# Run standup steps 0-9
./setup/standup.sh -v -c ${PWD}/scenarios/guides/pd-disaggregation2.sh -s 0-9
```

### How it works end-to-end

1. **Step 04** reads all files in `setup/preprocess/` (including the symlinked `llm-d-preflight-checks.py`) and creates the `llm-d-benchmark-preprocesses` ConfigMap in the target namespace.

2. **Scenario volume config** mounts the ConfigMap at `/setup/preprocess` inside pods:
   ```yaml
   volumes:
   - name: preprocesses
     configMap:
       defaultMode: 0755
       name: llm-d-benchmark-preprocesses
   volumeMounts:
   - name: preprocesses
     mountPath: /setup/preprocess
   ```

3. **Pod startup command** is generated via `REPLACE_ENV_*` substitution in the pod spec. The scenario's `EXTRA_ARGS` template uses `&&` between the preflight script and vllm:
   ```
   python3 /setup/preprocess/set_llmdbench_environment.py; \
   source $HOME/llmdbench_env.sh; \
   python3 /setup/preprocess/llm-d-preflight-checks.py && \
   vllm serve <model> --port $VLLM_INFERENCE_PORT ...
   ```

   The `&&` operator ensures that vllm only starts if the preflight checks script exits with code 0. If the script exits with a non-zero code (e.g., a future check detects a fatal misconfiguration), vllm will **not** start and the pod will fail — making the problem visible immediately rather than leading to hard-to-debug runtime errors.

   This is configured in the scenario's `LLMDBENCH_VLLM_MODELSERVICE_PREFILL_EXTRA_ARGS` and `LLMDBENCH_VLLM_MODELSERVICE_DECODE_EXTRA_ARGS` templates:
   ```bash
   cat << EOF > $LLMDBENCH_VLLM_MODELSERVICE_PREFILL_EXTRA_ARGS
   REPLACE_ENV_LLMDBENCH_VLLM_MODELSERVICE_PREFILL_PREPROCESS && \
   vllm serve /model-cache/models/REPLACE_ENV_LLMDBENCH_DEPLOY_CURRENT_MODEL \
   --host 0.0.0.0 \
   ...
   EOF
   ```

4. **Preflight script** checks `LLMD_PREFLIGHT_CHECKS` env var and either prints diagnostics and continues (default), or blocks with an HTTP server (`pause` mode) until `/exit` is called.

### Verifying the preflight output

After standup, check pod logs to see the preflight diagnostics:

```bash
kubectl logs -n <namespace> <pod-name> -c vllm | grep -A 50 "llm-d-preflight-checks.py starting"
```
