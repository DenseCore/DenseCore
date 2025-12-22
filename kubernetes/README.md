# Kubernetes Deployment

Deploy DenseCore on Kubernetes with High Availability.

## Quick Start

```bash
# Deploy all resources
kubectl apply -k .

# Check status
kubectl get pods -l app=densecore
```

## Configuration

Edit `configmap.yaml` or set environment variables:

- `MODEL_REPO`: HuggingFace repo ID (e.g., `Qwen/Qwen3-0.6B-GGUF`)
- `AUTH_ENABLED`: Set to `true` to require API keys.

## Architecture

- **Deployment**: Autoscaling (HPA) based on CPU usage.
- **Init Container**: Pre-downloads models from HuggingFace to PVC.
- **Service**: LoadBalancer exposing port 80/8080.
- **Monitoring**: Prometheus ServiceMonitor included.

## Secrets

Create a secret for API keys:

```bash
kubectl create secret generic densecore-secrets --from-literal=API_KEYS="sk-mykey:user:tier"
```
