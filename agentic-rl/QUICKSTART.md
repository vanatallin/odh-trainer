# Quick Start Guide

Get started with agentic RL training in 5 minutes!

## Prerequisites

- Kubernetes cluster (1.24+)
- Kubeflow Trainer v2 installed
- kubectl configured
- Docker installed
- Python 3.9+ (for SDK approach)
- At least 1 GPU node

## Choose Your Approach

### Option A: Python SDK (Recommended ✨)

Programmatic deployment with type safety and reusability.

#### Step 1: Clone and Navigate

```bash
git clone https://github.com/your-org/odh-trainer.git
cd odh-trainer/agentic-rl
```

#### Step 2: Install SDK Dependencies

```bash
cd sdk
pip install -r requirements.txt
```

#### Step 3: Build Images

```bash
cd ..
export REGISTRY=docker.io/your-username
make build
make push
```

#### Step 4: Deploy with Python

```bash
cd sdk
python trainjob_sdk.py \
  --mode simple \
  --student-image ${REGISTRY}/student-agent:latest \
  --reward-image ${REGISTRY}/reward-model:latest
```

**That's it!** See [sdk/README.md](sdk/README.md) for advanced examples.

---

### Option B: YAML + Make (Traditional)

Classic kubectl-based deployment.

#### Step 1: Clone and Navigate

```bash
git clone https://github.com/your-org/odh-trainer.git
cd odh-trainer/agentic-rl
```

## Step 2: Set Your Registry

```bash
export REGISTRY=docker.io/your-username
# Or use a private registry
# export REGISTRY=gcr.io/your-project
```

## Step 3: Build and Push Images

```bash
# Build both images
make build

# Push to registry
make push
```

This will build and push:
- `$REGISTRY/student-agent:latest`
- `$REGISTRY/reward-model:latest`

## Step 4: Deploy TrainingRuntime

```bash
make deploy-runtime
```

Verify:
```bash
kubectl get trainingruntime agentic-rl-pytorch
```

## Step 5: Deploy TrainJob

```bash
make deploy-job
```

Monitor:
```bash
# Watch job status
make status

# View logs
make logs

# Or view reward model logs
make logs-reward
```

## Step 6: Monitor Training

Check progress:
```bash
# Get job status
kubectl get trainjob agentic-rl-training

# Get pod status
kubectl get pods -l trainjob-name=agentic-rl-training

# View training metrics in logs
make logs | grep "Avg Reward"
```

## Step 7: Access Checkpoints

Once training completes, checkpoints are saved in the pod's `/checkpoints` directory.

To copy them locally:
```bash
POD=$(kubectl get pods -l trainjob-name=agentic-rl-training -o jsonpath='{.items[0].metadata.name}')
kubectl cp $POD:/checkpoints ./local-checkpoints -c node
```

## That's It!

You now have an agentic RL training job running on Kubernetes using Kubeflow Trainer v2.

## Next Steps

### Customize Training

Edit [manifests/train-job.yaml](manifests/train-job.yaml) to adjust:

```yaml
env:
  - name: NUM_EPOCHS
    value: "5"          # More epochs
  - name: BATCH_SIZE
    value: "8"          # Larger batches
  - name: LEARNING_RATE
    value: "2.0e-5"     # Different learning rate
```

Then redeploy:
```bash
make delete-job
make deploy-job
```

### Use Custom Prompts

```bash
kubectl apply -f examples/custom-prompts.yaml
```

### Scale to Multiple Nodes

```bash
kubectl apply -f examples/multi-node.yaml
```

### Use Persistent Storage

```bash
kubectl apply -f examples/with-persistent-storage.yaml
```

## Troubleshooting

### Job Not Starting

```bash
kubectl describe trainjob agentic-rl-training
kubectl get events --sort-by='.lastTimestamp'
```

### Pod Errors

```bash
POD=$(kubectl get pods -l trainjob-name=agentic-rl-training -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD
kubectl logs $POD -c node
kubectl logs $POD -c reward-model
```

### Out of Memory

Increase resources in [manifests/train-job.yaml](manifests/train-job.yaml):

```yaml
resourcesPerNode:
  requests:
    memory: "16Gi"  # Increase from 8Gi
```

For more troubleshooting, see [docs/troubleshooting.md](docs/troubleshooting.md).

## Clean Up

```bash
# Delete the job
make delete-job

# Delete the runtime
make delete-runtime
```

## Learn More

- [README.md](README.md) - Full documentation
- [docs/alternative-patterns.md](docs/alternative-patterns.md) - Deployment patterns
- [docs/scaling.md](docs/scaling.md) - Scaling strategies
- [docs/troubleshooting.md](docs/troubleshooting.md) - Common issues

## Getting Help

- [Open an issue](https://github.com/your-org/odh-trainer/issues)
- [Check existing issues](https://github.com/your-org/odh-trainer/issues)
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines

Happy training! 🚀
