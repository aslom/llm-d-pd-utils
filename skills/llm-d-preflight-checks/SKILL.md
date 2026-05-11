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
