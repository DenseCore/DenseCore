# DenseCore Kubernetes Deployment

This directory contains Kubernetes manifests for deploying DenseCore inference server.

## Quick Start

```bash
# 1. Update image name in kustomization.yaml
# Change 'yourname/densecore' to your Docker Hub username

# 2. Deploy all resources
kubectl apply -k .

# 3. Check status
kubectl get pods -l app.kubernetes.io/name=densecore
kubectl logs -f deployment/densecore
```

## Manifests

- `deployment.yaml` - Main deployment with probes and resource limits
- `service.yaml` - LoadBalancer service
- `configmap.yaml` - Configuration via environment variables
- `hpa.yaml` - Horizontal Pod Autoscaler
- `ingress.yaml` - Ingress for external access
- `servicemonitor.yaml` - Prometheus ServiceMonitor
- `kustomization.yaml` - Kustomize configuration
- `Dockerfile.downloader` - Model downloader image (init container)
- `download_model.py` - Python script for downloading models from HuggingFace

## Model Downloader Image

The deployment uses a dedicated init container to download models from HuggingFace Hub.
This image pre-bakes `huggingface_hub` to eliminate cold-start latency (~30s pip install).

### Building the Image

```bash
cd k8s
docker build -f Dockerfile.downloader -t densecore/downloader:latest .
docker push densecore/downloader:latest
```

### Features

- **Retry Logic**: Exponential backoff with 5 retries for network resilience
- **Existence Check**: Skips download if model already exists
- **Symlink Support**: Creates `main_model.gguf` symlink automatically
- **Small Footprint**: ~150MB (vs ~900MB for full Python image with deps)


## Configuration

### Environment Variables

Edit `configmap.yaml` to update configuration:

```bash
kubectl edit configmap densecore-config
kubectl rollout restart deployment densecore
```

### Secrets

Add API keys and HuggingFace token:

```bash
kubectl create secret generic densecore-secrets \
  --from-literal=API_KEYS="sk-key:user:tier" \
  --from-literal=HF_TOKEN="hf_xxx"
```

### Model Storage

Models are loaded from a PersistentVolumeClaim. Create one:

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

Upload your model:

```bash
# Create uploader pod
kubectl run uploader --image=busybox --rm -it --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"densecore-models-pvc"}}],"containers":[{"name":"uploader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"models","mountPath":"/models"}]}]}}'

# Copy model (in another terminal)
kubectl cp /path/to/model.gguf uploader:/models/main_model.gguf
```

## Scaling

### Manual Scaling

```bash
kubectl scale deployment densecore --replicas=3
```

### Auto Scaling

HPA is configured to scale between 1-10 replicas based on:
- CPU utilization > 70%
- Memory utilization > 80%

```bash
kubectl get hpa densecore-hpa
```

## Ingress

Configure external access by editing `ingress.yaml`:

1. Set your domain in `spec.rules[].host`
2. For TLS, uncomment the `tls` section and create a secret:

```bash
kubectl create secret tls densecore-tls \
  --cert=/path/to/tls.crt \
  --key=/path/to/tls.key
```

## Monitoring

Deploy Prometheus Operator, then apply ServiceMonitor:

```bash
kubectl apply -f servicemonitor.yaml
```

Metrics are available at `http://densecore-service/metrics`

## Troubleshooting

**Pods not starting:**
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

**Model not loading:**
- Verify PVC is bound: `kubectl get pvc`
- Check model path in deployment
- Verify file exists in PVC

**Service not accessible:**
```bash
kubectl get svc densecore-service
kubectl port-forward svc/densecore-service 8080:80
```

## Production Recommendations

1. Use specific image tags (not `latest`)
2. Enable authentication (`AUTH_ENABLED=true`)
3. Configure resource limits based on model size
4. Use node affinity for CPU-optimized nodes
5. Set up Ingress with TLS
6. Enable HPA for auto-scaling
7. Configure Prometheus monitoring
8. Use secrets for sensitive data
