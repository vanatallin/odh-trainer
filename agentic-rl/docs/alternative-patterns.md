# Alternative Deployment Patterns

This document describes alternative patterns for deploying the reward model beyond the sidecar approach.

## Pattern 1: Separate Service Deployment

### Overview

Deploy the reward model as a standalone Kubernetes Service that multiple training jobs can share.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  TrainJob Pod 1 │     │  TrainJob Pod 2 │     │  TrainJob Pod 3 │
│                 │     │                 │     │                 │
│  Student Agent  │────▶│  Student Agent  │────▶│  Student Agent  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Reward Model Service  │
                    │  (Deployment + Service)│
                    │    Multiple Replicas   │
                    └────────────────────────┘
```

### Advantages

- **Resource Efficiency**: One reward model serves multiple training jobs
- **Independent Scaling**: Scale reward model replicas independently
- **Persistence**: Reward model survives training job lifecycle
- **Updates**: Update reward model without restarting training
- **Cost**: Fewer GPU resources needed overall

### Disadvantages

- **Latency**: Network overhead for each reward query
- **Complexity**: Additional Kubernetes resources to manage
- **Availability**: Training depends on service availability

### Implementation

#### Step 1: Deploy Reward Model Service

Create `reward-model-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reward-model
  namespace: default
spec:
  replicas: 2  # Scale based on load
  selector:
    matchLabels:
      app: reward-model
  template:
    metadata:
      labels:
        app: reward-model
    spec:
      containers:
        - name: reward-model
          image: <your-registry>/reward-model:latest
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: PORT
              value: "8080"
            - name: MODEL_NAME
              value: "distilbert-base-uncased"
            - name: DEVICE
              value: "cuda"
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: "4Gi"
              cpu: "2"
            limits:
              nvidia.com/gpu: 1
              memory: "4Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: reward-model
  namespace: default
spec:
  selector:
    app: reward-model
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
  type: ClusterIP
```

Deploy:

```bash
kubectl apply -f reward-model-deployment.yaml
```

#### Step 2: Modify TrainingRuntime

Remove the reward model sidecar and use the service URL:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainingRuntime
metadata:
  name: agentic-rl-pytorch-service
  namespace: default
spec:
  mlPolicy:
    numNodes: 1
    torch:
      numProcPerNode: auto
  template:
    spec:
      replicatedJobs:
        - name: node
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: trainer
            spec:
              backoffLimit: 2
              template:
                spec:
                  containers:
                    - name: node
                      image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
                      command:
                        - python
                        - /workspace/train.py
                      env:
                        # Point to the service
                        - name: REWARD_MODEL_URL
                          value: "http://reward-model.default.svc.cluster.local:8080"
                        - name: PYTHONUNBUFFERED
                          value: "1"
                      resources:
                        requests:
                          nvidia.com/gpu: 1
                          memory: "8Gi"
                          cpu: "4"
                        limits:
                          nvidia.com/gpu: 1
                          memory: "8Gi"
                  restartPolicy: Never
```

#### Step 3: Deploy TrainJob

```bash
kubectl apply -f manifests/training-runtime-service.yaml
kubectl apply -f manifests/train-job.yaml
```

### Monitoring Service Load

Monitor the reward model service:

```bash
# Check pod status
kubectl get pods -l app=reward-model

# View logs
kubectl logs -l app=reward-model --tail=100

# Check service endpoints
kubectl get endpoints reward-model
```

### Scaling

Scale the reward model based on load:

```bash
# Manual scaling
kubectl scale deployment reward-model --replicas=5

# Or use HPA (Horizontal Pod Autoscaler)
kubectl autoscale deployment reward-model \
  --min=2 --max=10 \
  --cpu-percent=70
```

## Pattern 2: InitContainer + Shared Volume

### Overview

Use an InitContainer to download/prepare the reward model, then use a sidecar to serve it.

### Implementation

```yaml
spec:
  template:
    spec:
      initContainers:
        - name: download-reward-model
          image: busybox
          command:
            - sh
            - -c
            - |
              # Download reward model from S3/GCS/etc
              wget -O /models/reward-model.tar.gz https://example.com/reward-model.tar.gz
              tar -xzf /models/reward-model.tar.gz -C /models/
          volumeMounts:
            - name: model-cache
              mountPath: /models
      containers:
        - name: node
          # ... student container
        - name: reward-model
          # ... reward model sidecar
          volumeMounts:
            - name: model-cache
              mountPath: /models
              readOnly: true
      volumes:
        - name: model-cache
          emptyDir: {}
```

## Pattern 3: DaemonSet for Node-Local Reward Model

### Overview

Deploy reward model as a DaemonSet so each node has a local instance.

### Use Case

When you have many training pods per node and want to minimize network latency.

### Implementation

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: reward-model-daemonset
  namespace: default
spec:
  selector:
    matchLabels:
      app: reward-model-daemon
  template:
    metadata:
      labels:
        app: reward-model-daemon
    spec:
      hostNetwork: true  # Use host network for minimal latency
      containers:
        - name: reward-model
          image: <your-registry>/reward-model:latest
          ports:
            - containerPort: 8080
              hostPort: 8080
          # ... rest of config
      nodeSelector:
        node-type: gpu  # Only on GPU nodes
```

Training pods access via:
- `http://<node-ip>:8080` or
- `http://localhost:8080` if using `hostNetwork: true`

## Comparison Table

| Pattern | Latency | Resource Efficiency | Complexity | Best For |
|---------|---------|-------------------|------------|----------|
| Sidecar | Lowest (localhost) | Low (1:1 coupling) | Simple | Single training jobs, tight coupling needed |
| Service | Medium (cluster network) | High (shared) | Medium | Multiple concurrent training jobs |
| InitContainer | Low (localhost) | Medium | Medium | Custom model loading, version control |
| DaemonSet | Low (node-local) | Medium | High | Many pods per node, predictable scaling |

## Recommendations

1. **Development/Testing**: Use Sidecar (simplest)
2. **Production with Multiple Jobs**: Use Service (most efficient)
3. **High Throughput**: Use DaemonSet (lowest latency at scale)
4. **Custom Models**: Use InitContainer + Sidecar (flexibility)

## Migration Guide

### From Sidecar to Service

1. Deploy the reward model service
2. Update TrainingRuntime to remove sidecar
3. Update `REWARD_MODEL_URL` to point to service
4. No code changes needed in student or reward model

### From Service to Sidecar

1. Update TrainingRuntime to add sidecar container
2. Update `REWARD_MODEL_URL` to `http://localhost:8080`
3. Delete the service deployment
4. No code changes needed

## Performance Considerations

### Network Latency Impact

For a typical RL training loop:
- Sidecar: ~0.5ms per reward query
- Service (same zone): ~2-5ms per reward query
- Service (cross-zone): ~10-20ms per reward query

With 1000 reward queries per batch:
- Sidecar: ~0.5s overhead
- Service: ~2-20s overhead

**Mitigation**: Batch reward queries or use async requests.

### Batched Reward Queries

Modify reward model to accept batch requests:

```python
# server.py
def do_POST(self):
    if self.path == '/score_batch':
        data = json.loads(self.rfile.read())
        prompts = data['prompts']
        responses = data['responses']

        scores = [self.reward_model.score(p, r)
                  for p, r in zip(prompts, responses)]

        self._send_json({"scores": scores})
```

This reduces network round-trips significantly.
