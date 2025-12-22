# DenseCore Helm Chart

Deploy DenseCore on Kubernetes using Helm.

## Install

```bash
helm install my-densecore ./charts/densecore
```

## Configuration

### Custom Model

```bash
helm install my-densecore ./charts/densecore \
  --set model.repoId="TheBloke/Llama-2-7B-Chat-GGUF" \
  --set model.filename="llama-2-7b-chat.Q4_K_M.gguf"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image.repository` | `densecore/densecore` | Docker image name |
| `model.source` | `huggingface` | `huggingface`, `pvc`, or `hostPath` |
| `resources.limits.memory` | `8Gi` | Memory limit |
| `autoscaling.enabled` | `false` | Enable HPA |

See `values.yaml` for full configuration.
