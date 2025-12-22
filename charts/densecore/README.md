# DenseCore Helm Chart

Deploy the [DenseCore](https://github.com/Jake-Network/DenseCore) inference engine on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+

## Installing the Chart

To install the chart with the release name `my-densecore`:

```console
$ helm install my-densecore ./charts/densecore
```

This commands deploys DenseCore on the Kubernetes cluster in the default configuration. The [Parameters](#parameters) section lists the parameters that can be configured during installation.

> **Tip**: List all releases using `helm list`

## Model Configuration

By default, the chart downloads the `Qwen/Qwen2.5-0.5B-Instruct-GGUF` model from HuggingFace.

### Use a different model from HuggingFace

```console
$ helm install my-densecore ./charts/densecore \
  --set model.repoId="TheBloke/Llama-2-7B-Chat-GGUF" \
  --set model.filename="llama-2-7b-chat.Q4_K_M.gguf"
```

### Use a Persistent Volume (Recommended for Production)

1. Create a PVC containing your GGUF models.
2. Deploy the chart referencing the PVC:

```console
$ helm install my-densecore ./charts/densecore \
  --set model.source="pvc" \
  --set model.existingClaim="my-models-pvc" \
  --set model.filename="my-custom-model.gguf"
```

## Parameters

### Image
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| image.repository | string | `densecore/densecore` | Image name |
| image.tag | string | `""` (AppVersion) | Image tag |
| image.pullPolicy | string | `IfNotPresent` | Image pull policy |

### Model
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| model.source | string | `huggingface` | Source of model: `huggingface`, `pvc`, or `hostPath` |
| model.repoId | string | `Qwen/...` | HuggingFace Repository ID |
| model.filename | string | `qwen...gguf` | GGUF filename inside the repository |

### Resources
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| resources.limits.cpu | string | `4` | CPU Limit (4 vCPU recommended) |
| resources.limits.memory | string | `8Gi` | Memory Limit (8GB recommended) |

### Autoscaling (HPA)
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| autoscaling.enabled | bool | `false` | Enable Horizontal Pod Autoscaler |
| autoscaling.minReplicas | int | `1` | Min replicas |
| autoscaling.maxReplicas | int | `10` | Max replicas |
