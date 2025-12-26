# DenseCore Helm Chart

Deploy the [DenseCore](https://github.com/Jake-Network/DenseCore) inference engine on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- (Optional) Prometheus Operator for ServiceMonitor
- (Optional) KEDA for advanced autoscaling

## Installing the Chart

### From OCI Registry (Recommended)

```bash
helm install densecore oci://ghcr.io/jake-network/charts/densecore --version 0.3.0
```

### From Source

```bash
helm install densecore ./charts/densecore
```

> **Tip**: List all releases using `helm list`

## Quick Start

```bash
# Deploy with default model (Qwen 0.5B)
helm install densecore ./charts/densecore

# Deploy with custom model
helm install densecore ./charts/densecore \
  --set model.repoId="TheBloke/Llama-2-7B-Chat-GGUF" \
  --set model.filename="llama-2-7b-chat.Q4_K_M.gguf"

# Deploy with existing PVC
helm install densecore ./charts/densecore \
  --set model.source="pvc" \
  --set model.existingClaim="my-models-pvc"
```

## Model Configuration

### HuggingFace (Default)

The chart automatically downloads models from HuggingFace using an init container:

```yaml
model:
  source: huggingface
  repoId: "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
  filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf"
  # For gated models:
  hfTokenSecret: "hf-token-secret"  # Secret containing HF_TOKEN key
```

### Persistent Volume (Production)

For production, use a PVC to avoid repeated downloads:

```bash
# Deploy with existing PVC - specify the exact filename in your PVC
helm install densecore ./charts/densecore \
  --set model.source="pvc" \
  --set model.existingClaim="my-models-pvc" \
  --set model.filename="llama-2-7b-chat.Q4_K_M.gguf"
```

> **Important**: `model.filename` must match the actual filename inside your PVC.  
> The full path will be `{model.path}/{model.filename}` (default: `/models/llama-2-7b-chat.Q4_K_M.gguf`).

```yaml
# values.yaml example
model:
  source: pvc
  existingClaim: "my-models-pvc"  # Pre-existing PVC with your model
  filename: "my-custom-model.gguf"  # Required: exact filename in PVC
  path: "/models"  # Mount path (default)
  
  # Or create a new PVC:
  pvc:
    storageClassName: "gp3"
    size: "50Gi"
    accessModes:
      - ReadWriteMany
```

### Host Path (Development)

```yaml
model:
  source: hostPath
  hostPath: "/mnt/models"
  filename: "my-model.gguf"  # Required: exact filename in hostPath
```

## Parameters

### Image

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `image.repository` | string | `densecore/densecore` | Image name |
| `image.tag` | string | `""` (AppVersion) | Image tag |
| `image.pullPolicy` | string | `IfNotPresent` | Image pull policy |

### Model

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model.source` | string | `huggingface` | Source: `huggingface`, `pvc`, `hostPath`, `emptyDir` |
| `model.repoId` | string | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | HuggingFace Repository ID |
| `model.filename` | string | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | GGUF filename |
| `model.hfTokenSecret` | string | `""` | Secret name containing `HF_TOKEN` |
| `model.path` | string | `/models` | Mount path inside container |

### Resources

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `resources.requests.cpu` | string | `2` | CPU request |
| `resources.requests.memory` | string | `4Gi` | Memory request |
| `resources.limits.cpu` | string | `8` | CPU limit |
| `resources.limits.memory` | string | `16Gi` | Memory limit |

### Probes

Fully configurable health probes via `values.yaml`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `probes.liveness.path` | string | `/health/live` | Liveness endpoint |
| `probes.liveness.initialDelaySeconds` | int | `30` | Initial delay |
| `probes.liveness.periodSeconds` | int | `10` | Check interval |
| `probes.liveness.timeoutSeconds` | int | `5` | Timeout |
| `probes.liveness.failureThreshold` | int | `3` | Failure threshold |
| `probes.readiness.path` | string | `/health/ready` | Readiness endpoint |
| `probes.readiness.initialDelaySeconds` | int | `10` | Initial delay |
| `probes.startup.path` | string | `/health/startup` | Startup endpoint |
| `probes.startup.failureThreshold` | int | `60` | Allow 5min for model loading |

### Autoscaling (HPA)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `autoscaling.enabled` | bool | `false` | Enable HPA |
| `autoscaling.minReplicas` | int | `1` | Minimum replicas |
| `autoscaling.maxReplicas` | int | `10` | Maximum replicas |
| `autoscaling.targetCPUUtilizationPercentage` | int | `70` | Target CPU % |
| `autoscaling.targetMemoryUtilizationPercentage` | int | `80` | Target Memory % |
| `autoscaling.behavior` | object | See values.yaml | HPA scale up/down behavior |

**Advanced HPA Behavior:**

```yaml
autoscaling:
  enabled: true
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Pods
          value: 4
          periodSeconds: 60
```

### KEDA (Advanced Autoscaling)

Scale based on inference queue depth instead of CPU:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `keda.enabled` | bool | `false` | Enable KEDA ScaledObject |
| `keda.minReplicaCount` | int | `1` | Minimum replicas |
| `keda.maxReplicaCount` | int | `10` | Maximum replicas |
| `keda.prometheusServerAddress` | string | `http://prometheus-server...` | Prometheus URL |
| `keda.triggers.pendingRequests.threshold` | string | `5` | Scale up threshold |
| `keda.triggers.activeRequests.threshold` | string | `8` | Scale up threshold |

### Redis (Distributed State)

Enable Redis for multi-replica deployments:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `redis.enabled` | bool | `false` | Enable Redis integration |
| `redis.url` | string | `redis://redis:6379` | Redis URL |
| `redis.existingSecret` | string | `""` | Secret for password |
| `redis.rateLimit.enabled` | bool | `true` | Distributed rate limiting |
| `redis.keyStore.enabled` | bool | `true` | Distributed API key storage |

### Monitoring

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `serviceMonitor.enabled` | bool | `false` | Create ServiceMonitor (Prometheus Operator) |
| `serviceMonitor.interval` | string | `15s` | Scrape interval |
| `serviceMonitor.labels` | object | `{}` | Additional labels (e.g., `release: prometheus`) |

### Network Policy

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `networkPolicy.enabled` | bool | `false` | Enable NetworkPolicy |
| `networkPolicy.allowIngressController` | bool | `true` | Allow traffic from ingress |

### Pod Disruption Budget

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `podDisruptionBudget.enabled` | bool | `false` | Enable PDB |
| `podDisruptionBudget.minAvailable` | int | `1` | Minimum available pods |

## Examples

### Production Deployment

```yaml
# production-values.yaml
replicaCount: 3

model:
  source: pvc
  existingClaim: production-models

resources:
  requests:
    cpu: "4"
    memory: "8Gi"
  limits:
    cpu: "16"
    memory: "32Gi"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300

redis:
  enabled: true
  url: "redis://redis-master.redis.svc:6379"
  existingSecret: "redis-password"

serviceMonitor:
  enabled: true
  labels:
    release: prometheus

networkPolicy:
  enabled: true

podDisruptionBudget:
  enabled: true
  minAvailable: 2
```

```bash
helm install densecore ./charts/densecore -f production-values.yaml
```

### Development with Ingress

```yaml
# dev-values.yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: densecore.dev.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: densecore-tls
      hosts:
        - densecore.dev.example.com
```

## Upgrading

```bash
helm upgrade densecore ./charts/densecore -f values.yaml
```

## Uninstalling

```bash
helm uninstall densecore
```

## Troubleshooting

### Model Download Fails

Check init container logs:

```bash
kubectl logs <pod-name> -c model-downloader
```

### Pod Stuck in Starting

The startup probe allows 5 minutes (60 × 5s) for model loading. For larger models, increase:

```yaml
probes:
  startup:
    failureThreshold: 120  # 10 minutes
```

### OOMKilled

Increase memory limits:

```yaml
resources:
  limits:
    memory: "32Gi"
```

## See Also

- [Cloud-Native Guide](../docs/CLOUD_NATIVE.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)
- [API Reference](../docs/API_REFERENCE.md)
