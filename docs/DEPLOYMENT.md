# Deployment Guide

Production deployment guide for DenseCore across Docker, Kubernetes, and cloud platforms.

---

## Table of Contents

- [Docker Deployment](#docker-deployment)
- [Docker Compose](#docker-compose)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Environment Variables](#environment-variables)
- [Health Checks & Monitoring](#health-checks--monitoring)
- [Authentication](#authentication)
- [Production Best Practices](#production-best-practices)
- [Troubleshooting](#troubleshooting)

---

Complete guide for deploying DenseCore Go server via Docker and Kubernetes.

## Docker Deployment

### Quick Start

Pull the official Docker image from GitHub Container Registry:

```bash
docker pull ghcr.io/jake-network/densecore:latest
```

Or build from source:

```bash
cd /path/to/DenseCore
docker build -t ghcr.io/jake-network/densecore:latest .
```

### Running the Container

**Basic usage (no model - health check only):**
```bash
docker run -p 8080:8080 ghcr.io/jake-network/densecore:latest
```

**With a model from local directory:**
```bash
docker run -p 8080:8080 \
  -v /path/to/models:/models:ro \
  -e MAIN_MODEL_PATH=/models/qwen2.5-0.5b-q4.gguf \
  ghcr.io/jake-network/densecore:latest
```

**With authentication (production):**
```bash
docker run -p 8080:8080 \
  -v /path/to/models:/models:ro \
  -e MAIN_MODEL_PATH=/models/qwen2.5-0.5b-q4.gguf \
  -e AUTH_ENABLED=true \
  -e API_KEYS="sk-prod-key:user1:enterprise,sk-dev-key:user2:free" \
  ghcr.io/jake-network/densecore:latest
```

> [!CAUTION]
> **SECURITY WARNING**: Do **NOT** use the example keys (`sk-prod-key...`) in production. These are public examples. Generate strong, random keys and inject them via Kubernetes Secrets or a secure environment variable manager.

### 3. Test the Server

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Chat completion (if model is loaded)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Docker Compose Deployment

For a complete stack with monitoring (optional):

```bash
cd /path/to/DenseCore
docker-compose up -d
```

This will start:
- DenseCore server on port 8080
- (Optional) Prometheus on port 9090
- (Optional) Grafana on port 3000

**View logs:**
```bash
docker-compose logs -f densecore
```

**Stop services:**
```bash
docker-compose down
```

---

## Building Your Own Image

### Prerequisites
- Docker 20.10+
- 8GB RAM minimum
- 10GB free disk space

### Build Steps

```bash
cd /path/to/DenseCore

# Build the image
docker build -t yourname/densecore:latest .

# Tag for versioning
docker tag yourname/densecore:latest yourname/densecore:v2.0.0

# Push to Docker Hub
docker push yourname/densecore:latest
docker push yourname/densecore:v2.0.0
```

**Build with custom parameters:**
```bash
docker build \
  --build-arg VERSION=2.0.0 \
  -t yourname/densecore:v2.0.0 \
  .
```

---

## Kubernetes Deployment

### Option 1: Helm Chart (Recommended)

Helm is the easiest way to manage DenseCore on Kubernetes.

**Prerequisites:**
- Helm 3.0+
- Kubernetes 1.19+

**Install from source:**

```bash
git clone https://github.com/Jake-Network/DenseCore.git
cd DenseCore

# Install with default settings (downloads Qwen2.5-0.5B from HF)
helm install densecore ./charts/densecore
```

**Custom Installation:**

```bash
# Use a custom model from HuggingFace
helm install densecore ./charts/densecore \
  --set model.repoId="TheBloke/Llama-2-7B-Chat-GGUF" \
  --set model.filename="llama-2-7b-chat.Q4_K_M.gguf"

# Enable Autoscaling (HPA)
helm install densecore ./charts/densecore \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=10
```

See [charts/densecore/README.md](../charts/densecore/README.md) for all configuration options.

---

### Option 2: Manual Manifests (Advanced)

### Prerequisites
- Kubernetes 1.24+
- kubectl configured
- Storage for models (PVC)

### Deploy to Kubernetes

**Using kubectl:**
```bash
cd /path/to/DenseCore/k8s

# Update the image name in kustomization.yaml first
# Change: yourname/densecore to your actual Docker Hub username

# Apply all manifests
kubectl apply -k .

# Or apply individually
kubectl apply -f configmap.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
kubectl apply -f hpa.yaml
```

**Check deployment status:**
```bash
kubectl get pods -l app.kubernetes.io/name=densecore
kubectl logs -f deployment/densecore
```

**Access the service:**
```bash
# If using LoadBalancer
kubectl get svc densecore-service

# If using port-forward for testing
kubectl port-forward svc/densecore-service 8080:80
curl http://localhost:8080/health
```

### Configure Model Storage

**Create PVC for models:**
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: densecore-models-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard
EOF
```

**Upload model to PVC:**
```bash
# Create a temporary pod to upload
kubectl run uploader --image=busybox --rm -it --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"densecore-models-pvc"}}],"containers":[{"name":"uploader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"models","mountPath":"/models"}]}]}}'

# In another terminal, copy model
kubectl cp /path/to/model.gguf uploader:/models/main_model.gguf

# Delete the uploader pod
kubectl delete pod uploader
```

### Scale the Deployment

**Manual scaling:**
```bash
kubectl scale deployment densecore --replicas=3
```

**Using HPA (auto-scaling - CPU-based):**
The HPA is already configured in `k8s/hpa.yaml` and will automatically scale based on CPU/memory usage.

```bash
# Check HPA status
kubectl get hpa densecore-hpa
```

### KEDA Queue-Based Autoscaling (Recommended)

For LLM inference, **CPU-based HPA is a lagging indicator**. KEDA scales based on **pending request queue depth** (leading indicator):

```bash
# Install KEDA
helm install keda kedacore/keda --namespace keda --create-namespace

# Apply KEDA ScaledObject (instead of HPA)
kubectl delete -f k8s/hpa.yaml  # Remove CPU-based HPA
kubectl apply -f k8s/keda-scaledobject.yaml
```

**How it works:**
- Prometheus scrapes `densecore_pending_requests` metric from `/metrics`
- KEDA scales when queue depth exceeds threshold (5 per replica)
- Scales **before** latency spikes, not after

**Configuration (`k8s/keda-scaledobject.yaml`):**
```yaml
triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server.monitoring.svc:80
      query: sum(densecore_pending_requests{app="densecore"})
      threshold: "5"  # Scale at 5 pending requests per replica
```

### Update Configuration

**Update ConfigMap:**
```bash
kubectl edit configmap densecore-config
# Or
kubectl apply -f k8s/configmap.yaml
```

**Restart pods to pick up changes:**
```bash
kubectl rollout restart deployment densecore
```

---

## Environment Variables Reference

### Required
- `MAIN_MODEL_PATH` - Path to GGUF model file (e.g., `/models/model.gguf`)

### Optional
- `PORT` - Server port (default: `8080`)
- `HOST` - Bind address (default: `0.0.0.0`)
- `THREADS` - CPU threads (default: `0` = auto-detect)
- `AUTH_ENABLED` - Enable API key authentication (`true`/`false`)
- `API_KEYS` - Comma-separated API keys (`key:user:tier,key:user:tier`)
- `RATE_LIMIT_ENABLED` - Enable rate limiting (default: `true`)
- `RATE_LIMIT_RPS` - Requests per second (default: `100`)
- `LOG_FORMAT` - Log format: `json` or `text` (default: `json`)

### Timeouts
- `READ_TIMEOUT` - HTTP read timeout (default: `30s`)
- `WRITE_TIMEOUT` - HTTP write timeout (default: `120s`)
- `SHUTDOWN_TIMEOUT` - Graceful shutdown timeout (default: `30s`)

### CORS
- `CORS_ALLOWED_ORIGINS` - Allowed origins (default: `*`)

---

## Health Checks

DenseCore provides multiple health endpoints for Kubernetes probes:

- `/health` - Overall health status
- `/health/live` - Liveness probe (is the server running?)
- `/health/ready` - Readiness probe (is it ready to serve traffic?)
- `/health/startup` - Startup probe (has it finished initializing?)

> [!NOTE]
> **Readiness Probe Timeout**: Support for `readinessProbe` is fully configured, but be aware that the initial Model Load in C++ can be **blocking**. If loading a large model (e.g., 70B), the probe might time out or fail until the model is fully resident in RAM. Adjust your `initialDelaySeconds` and `failureThreshold` accordingly.

---

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` endpoint in Prometheus format.

**Key metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `model_inference_duration_seconds` - Inference time
- `active_requests` - Currently processing requests

### Grafana Dashboards

If using docker-compose with Grafana:
1. Access Grafana at http://localhost:3000
2. Login with `admin/admin`
3. Import the DenseCore dashboard from `/grafana/densecore-dashboard.json`

---

## Troubleshooting

### Container won't start

**Check logs:**
```bash
docker logs <container-id>
# or
kubectl logs deployment/densecore
```

**Common issues:**
- Model file not found: Verify `MAIN_MODEL_PATH` and volume mount
- Out of memory: Reduce model size or increase container RAM
- Permission denied: Check file permissions on model directory

### Server is slow

- Increase `THREADS` environment variable
- Allocate more CPU resources in Kubernetes
- Use quantized models (INT4/INT8)

### Authentication not working

- Verify `AUTH_ENABLED=true`
- Check `API_KEYS` format: `key:user:tier`
- Include API key in request header: `Authorization: Bearer sk-xxx`

### Kubernetes deployment fails

**Check pod status:**
```bash
kubectl describe pod <pod-name>
```

**Common issues:**
- ImagePullBackOff: Update image name in deployment or kustomization
- CrashLoopBackOff: Check logs for errors
- PVC not bound: Verify storage class and PVC exists

---

## Authentication

DenseCore supports API key authentication for production deployments.

### Enabling Authentication

```bash
docker run -p 8080:8080 \
  -e AUTH_ENABLED=true \
  -e API_KEYS="sk-prod-key:user1:enterprise,sk-dev-key:user2:free" \
  ghcr.io/jake-network/densecore:latest
```

### Using API Keys

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-prod-key" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### API Key Format

`API_KEYS` environment variable format: `key:user_id:tier,key2:user2:tier2`

- **key**: The secret API key (e.g., `sk-prod-abc123`)
- **user_id**: User identifier for logging
- **tier**: Rate limit tier (`free`, `pro`, `enterprise`)

### Security Best Practices

- Use Kubernetes Secrets or AWS Secrets Manager for API keys
- Enable HTTPS/TLS in production (use an ingress controller)
- Rotate keys regularly
- Monitor authentication failures

---

## Production Best Practices

1. **Use specific image tags** - Don't use `latest` in production
2. **Enable authentication** - Set `AUTH_ENABLED=true`
3. **Configure resource limits** - Set appropriate CPU/memory limits
4. **Use secrets for API keys** - Never commit secrets to git
5. **Enable monitoring** - Deploy Prometheus and Grafana
6. **Set up ingress with TLS** - Use cert-manager for SSL
7. **Regular backups** - Backup your models and configuration
8. **Use node affinity** - Schedule pods on CPU-optimized nodes

---

## Next Steps

- Configure Ingress for external access
- Set up SSL/TLS certificates
- Configure horizontal pod autoscaling
- Set up centralized logging (ELK stack)
- Implement CI/CD pipeline for automated deployments
