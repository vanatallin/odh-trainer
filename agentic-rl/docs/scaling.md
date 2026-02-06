# Scaling Guide

This guide covers scaling strategies for agentic RL training.

## Scaling Dimensions

### 1. Vertical Scaling (Bigger Resources)

Increase resources per pod to handle larger models or batch sizes.

#### Configuration

```yaml
spec:
  trainer:
    resourcesPerNode:
      requests:
        nvidia.com/gpu: 4      # Use 4 GPUs per pod
        memory: "32Gi"         # Increase memory
        cpu: "16"              # More CPUs
      limits:
        nvidia.com/gpu: 4
        memory: "32Gi"
```

#### When to Use
- Large language models (7B+ parameters)
- Large batch sizes
- High-resolution observations

#### Considerations
- Node must have sufficient resources
- Cost increases linearly
- May hit single-node limits

### 2. Horizontal Scaling (More Pods)

Distribute training across multiple nodes using data parallelism.

#### Configuration

```yaml
spec:
  trainer:
    numNodes: 4                # 4 training nodes
    numProcPerNode: 2          # 2 GPUs per node
    resourcesPerNode:
      requests:
        nvidia.com/gpu: 2
```

#### Implementation

The Kubeflow Trainer v2 automatically sets up distributed training using PyTorch DDP.

Environment variables are automatically set:
- `RANK`: Global rank of the process
- `WORLD_SIZE`: Total number of processes
- `LOCAL_RANK`: Rank within the node
- `MASTER_ADDR`: Address of rank 0
- `MASTER_PORT`: Port for communication

#### When to Use
- Large datasets requiring more throughput
- Faster training needed
- Experimenting with hyperparameters in parallel

#### Considerations
- Communication overhead between nodes
- Each node needs its own reward model sidecar
- Requires distributed training code modifications

### 3. Reward Model Scaling

Scale the reward model independently from training.

#### Pattern A: Sidecar (1:1)

Each training pod has its own reward model sidecar.

**Pros**: Lowest latency, no network overhead
**Cons**: More GPU resources needed

```yaml
# Each pod has both containers
spec:
  trainer:
    numNodes: 4  # = 4 student + 4 reward model containers
```

#### Pattern B: Shared Service

Multiple training pods share a reward model service.

**Pros**: Resource efficient, independent scaling
**Cons**: Network latency, potential bottleneck

```yaml
# Reward model deployment (separate)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reward-model
spec:
  replicas: 3  # Scale based on load
```

See [alternative-patterns.md](alternative-patterns.md) for details.

## Distributed PPO

### Synchronous PPO (Recommended)

All workers collect trajectories, then synchronize for PPO updates.

#### Architecture

```
Node 0              Node 1              Node 2
┌──────────┐        ┌──────────┐        ┌──────────┐
│ Student  │        │ Student  │        │ Student  │
│ + Reward │        │ + Reward │        │ + Reward │
└────┬─────┘        └────┬─────┘        └────┬─────┘
     │                   │                   │
     └───────────────────┼───────────────────┘
                         │
                    Synchronize
                    Gradients
```

#### Implementation

Modify `train.py` to use distributed training:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    return local_rank

def main():
    # Setup
    local_rank = setup_distributed()

    # Wrap model with DDP
    agent.model = DDP(
        agent.model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    # Training loop
    for epoch in range(num_epochs):
        # Each worker generates trajectories
        trajectories = collect_trajectories()

        # Synchronize gradients across workers
        loss.backward()

        # DDP automatically averages gradients
        optimizer.step()

        # Synchronize metrics
        if dist.get_rank() == 0:
            # Only rank 0 logs
            logger.info(f"Epoch {epoch} complete")
```

#### Configuration

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: distributed-ppo
spec:
  runtimeRef:
    name: agentic-rl-pytorch
  trainer:
    numNodes: 4
    numProcPerNode: 2  # 2 GPUs per node
    env:
      - name: NCCL_DEBUG
        value: "INFO"  # For debugging communication
```

### Asynchronous PPO

Workers update policy independently without synchronization.

**Pros**: Higher throughput, more exploration
**Cons**: Less stable, may diverge

Not recommended for initial implementation.

## Multi-Node Considerations

### 1. Network Requirements

- **High bandwidth**: 10+ Gbps recommended
- **Low latency**: InfiniBand or high-speed Ethernet
- **Topology**: Nodes in same availability zone

### 2. Storage

Use shared storage for checkpoints:

```yaml
spec:
  template:
    spec:
      volumes:
        - name: checkpoints
          persistentVolumeClaim:
            claimName: training-checkpoints
```

Create PVC:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-checkpoints
spec:
  accessModes:
    - ReadWriteMany  # Multi-node access
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs  # Use network storage
```

### 3. Synchronization

Ensure all workers synchronize:

```python
# Before checkpoint save
dist.barrier()

if dist.get_rank() == 0:
    # Only rank 0 saves
    agent.save_checkpoint(path)

# Wait for save to complete
dist.barrier()
```

### 4. Reward Model Per Node

Each node should have access to a reward model:

**Option A**: Sidecar per pod (simple)
```yaml
# Each pod gets its own reward model sidecar
# Already configured in training-runtime.yaml
```

**Option B**: DaemonSet (efficient)
```yaml
# One reward model per node, shared by all pods on that node
# See alternative-patterns.md for DaemonSet example
```

## Performance Benchmarks

### Single Node Baseline

| Configuration | Throughput | Cost/Hour |
|---------------|------------|-----------|
| 1 x A100 40GB | 100 episodes/hr | $3 |
| 2 x A100 40GB | 180 episodes/hr | $6 |
| 4 x A100 40GB | 320 episodes/hr | $12 |

### Multi-Node Scaling

| Nodes | GPUs | Throughput | Scaling Efficiency |
|-------|------|------------|-------------------|
| 1 | 4 | 320 ep/hr | 100% |
| 2 | 8 | 600 ep/hr | 93% |
| 4 | 16 | 1100 ep/hr | 86% |
| 8 | 32 | 2000 ep/hr | 78% |

*Note: Efficiency drops due to communication overhead*

## Optimization Strategies

### 1. Gradient Accumulation

Simulate larger batch sizes without more memory:

```python
# In train.py
accumulation_steps = 4

for i, batch in enumerate(batches):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision Training

Use FP16 to reduce memory and increase speed:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = compute_loss(outputs)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Batch Reward Queries

Reduce network overhead:

```python
# Collect all prompts and responses
batch_prompts = [...]
batch_responses = [...]

# Single batched request
rewards = agent.get_rewards_batch(batch_prompts, batch_responses)
```

Modify reward model server:

```python
# In server.py
def do_POST(self):
    if self.path == '/score_batch':
        data = json.loads(self.rfile.read())
        prompts = data['prompts']
        responses = data['responses']

        scores = []
        for p, r in zip(prompts, responses):
            scores.append(self.reward_model.score(p, r))

        self._send_json({"scores": scores})
```

### 4. Checkpoint Frequency

Balance between safety and performance:

```python
# Save every N steps instead of every epoch
if global_step % checkpoint_interval == 0:
    agent.save_checkpoint(path)
```

### 5. Async Logging

Don't block training for logging:

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

def log_metrics(metrics):
    # Send to W&B, TensorBoard, etc
    pass

# In training loop
executor.submit(log_metrics, current_metrics)
```

## Monitoring Distributed Training

### 1. Per-Node Metrics

Track metrics from each node:

```python
# In train.py
metrics = {
    'rank': dist.get_rank(),
    'node': os.environ.get('HOSTNAME'),
    'reward': avg_reward,
    'loss': avg_loss,
}

# Gather from all ranks
all_metrics = [None] * dist.get_world_size()
dist.all_gather_object(all_metrics, metrics)

if dist.get_rank() == 0:
    for m in all_metrics:
        logger.info(f"Node {m['node']}: reward={m['reward']}")
```

### 2. Communication Overhead

Monitor time spent in synchronization:

```python
import time

sync_start = time.time()
dist.barrier()
sync_time = time.time() - sync_start

logger.info(f"Synchronization took {sync_time:.3f}s")
```

### 3. GPU Utilization

Check if GPUs are being utilized:

```bash
# On each node
kubectl exec <pod-name> -- nvidia-smi dmon -s u -c 100
```

## Cost Optimization

### 1. Spot Instances

Use spot/preemptible instances for cost savings:

```yaml
spec:
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-preemptible: "true"
      # Or for AWS
      # nodeSelector:
      #   eks.amazonaws.com/capacityType: SPOT
```

Add checkpointing to handle interruptions:

```python
# Save checkpoints frequently
if global_step % 100 == 0:
    agent.save_checkpoint(checkpoint_path)

# Resume from checkpoint
if os.path.exists(checkpoint_path):
    agent.load_checkpoint(checkpoint_path)
```

### 2. Auto-Scaling

Scale down when not needed:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reward-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reward-model
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### 3. Resource Requests vs Limits

Set appropriate requests and limits:

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
    memory: "8Gi"    # What you need
    cpu: "4"
  limits:
    nvidia.com/gpu: 1
    memory: "12Gi"   # Maximum allowed
    # No CPU limit to avoid throttling
```

## Example: 4-Node Training Job

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: distributed-agentic-rl
  namespace: default
spec:
  runtimeRef:
    name: agentic-rl-pytorch
  trainer:
    numNodes: 4
    numProcPerNode: 2
    resourcesPerNode:
      requests:
        nvidia.com/gpu: 2
        memory: "16Gi"
        cpu: "8"
      limits:
        nvidia.com/gpu: 2
        memory: "16Gi"
    env:
      - name: NUM_EPOCHS
        value: "10"
      - name: BATCH_SIZE
        value: "8"
      - name: NCCL_DEBUG
        value: "INFO"
      - name: NCCL_SOCKET_IFNAME
        value: "eth0"
```

This creates:
- 4 pods (nodes)
- 2 GPUs per pod = 8 total GPUs
- 8 training processes total
- Each pod has its own reward model sidecar
