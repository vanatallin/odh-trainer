# Troubleshooting Guide

This guide helps you debug common issues with agentic RL training.

## Common Issues

### 1. Reward Model Not Ready

**Symptom**: Student container logs show:
```
Reward model not ready yet, retrying in 5s...
RuntimeError: Reward model at http://localhost:8080 did not become ready within 300s
```

**Causes**:
- Reward model container is still downloading the model
- Reward model crashed during initialization
- Port mismatch between containers

**Solutions**:

1. Check reward model logs:
```bash
kubectl logs <pod-name> -c reward-model
```

2. Increase readiness probe delay:
```yaml
readinessProbe:
  initialDelaySeconds: 60  # Increase from 30
```

3. Verify port configuration:
```bash
kubectl exec <pod-name> -c reward-model -- netstat -tuln | grep 8080
```

4. Check if model download is slow:
```bash
# Add download progress logging to server.py
logger.info(f"Downloading model {model_name}...")
```

### 2. CUDA Out of Memory

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Causes**:
- Both student and reward model sharing one GPU
- Model too large for available GPU memory
- Batch size too large

**Solutions**:

1. **Option A**: Use separate GPUs per container
```yaml
spec:
  containers:
    - name: node
      resources:
        limits:
          nvidia.com/gpu: 1  # GPU 0
    - name: reward-model
      resources:
        limits:
          nvidia.com/gpu: 1  # GPU 1
  nodeSelector:
    gpu-count: "2"  # Ensure node has 2+ GPUs
```

2. **Option B**: Use CPU for reward model
```yaml
- name: reward-model
  env:
    - name: DEVICE
      value: "cpu"
  resources:
    limits:
      # No GPU allocation
      memory: "4Gi"
```

3. **Option C**: Reduce batch size
```yaml
env:
  - name: BATCH_SIZE
    value: "2"  # Reduce from 4
```

4. **Option D**: Use gradient checkpointing
```python
# In agent.py
self.model.gradient_checkpointing_enable()
```

### 3. Training Hangs or Stalls

**Symptom**: Training stops progressing, no new logs

**Causes**:
- Deadlock waiting for reward model
- Network issues between containers
- GPU hang

**Solutions**:

1. Check if reward model is responding:
```bash
kubectl exec <pod-name> -c node -- curl http://localhost:8080/health
```

2. Add timeout to reward queries (already implemented in `agent.py`):
```python
response = requests.post(url, json=data, timeout=10)
```

3. Check for GPU issues:
```bash
kubectl exec <pod-name> -c node -- nvidia-smi
```

4. Add more verbose logging:
```python
# In train.py
logger.setLevel(logging.DEBUG)
```

### 4. Pod Gets OOMKilled

**Symptom**:
```
kubectl get pods
NAME                  READY   STATUS      RESTARTS
agentic-rl-xyz        1/2     OOMKilled   1
```

**Causes**:
- Container memory limit too low
- Memory leak in training code
- Large model gradients

**Solutions**:

1. Increase memory limits:
```yaml
resources:
  requests:
    memory: "16Gi"  # Increase from 8Gi
  limits:
    memory: "16Gi"
```

2. Monitor memory usage:
```bash
kubectl top pod <pod-name> --containers
```

3. Add memory cleanup in training loop:
```python
# In train.py
import gc
torch.cuda.empty_cache()
gc.collect()
```

### 5. Slow Training Speed

**Symptom**: Training is much slower than expected

**Causes**:
- Reward model latency
- Inefficient data loading
- GPU underutilization

**Solutions**:

1. Profile reward model latency:
```python
# Add timing in agent.py
start = time.time()
reward = self.get_reward(prompt, response)
logger.info(f"Reward query took {time.time() - start:.3f}s")
```

2. Batch reward queries:
```python
# Modify to query rewards in batch
rewards = []
for prompt, response in zip(prompts, responses):
    rewards.append(self.get_reward(prompt, response))
```

3. Use mixed precision training:
```python
# In agent.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = self.model(inputs)
```

4. Check GPU utilization:
```bash
kubectl exec <pod-name> -c node -- nvidia-smi dmon
```

### 6. Connection Refused to Reward Model

**Symptom**:
```
requests.exceptions.ConnectionError: Connection refused
```

**Causes**:
- Reward model not started yet
- Port mismatch
- Container networking issue

**Solutions**:

1. Verify both containers are in the same pod:
```bash
kubectl get pod <pod-name> -o jsonpath='{.spec.containers[*].name}'
```

2. Check port configuration:
```bash
kubectl get pod <pod-name> -o yaml | grep -A 5 ports
```

3. Test connectivity:
```bash
kubectl exec <pod-name> -c node -- nc -zv localhost 8080
```

### 7. Model Weights Not Updating

**Symptom**: Loss stays constant, rewards don't improve

**Causes**:
- Learning rate too low
- Gradients not flowing
- Optimizer not configured correctly

**Solutions**:

1. Check gradients:
```python
# In agent.py
for name, param in self.model.named_parameters():
    if param.grad is not None:
        logger.info(f"{name}: {param.grad.norm()}")
```

2. Increase learning rate:
```yaml
env:
  - name: LEARNING_RATE
    value: "1.0e-4"  # Increase from 1e-5
```

3. Verify optimizer:
```python
logger.info(f"Optimizer state: {self.optimizer.state_dict()}")
```

### 8. Checkpoints Not Saving

**Symptom**: No checkpoints in `/checkpoints` directory

**Causes**:
- Permission issues
- Volume not mounted
- Path mismatch

**Solutions**:

1. Check volume mount:
```bash
kubectl get pod <pod-name> -o yaml | grep -A 5 volumeMounts
```

2. Verify directory permissions:
```bash
kubectl exec <pod-name> -c node -- ls -la /checkpoints
```

3. Add error handling:
```python
try:
    agent.save_checkpoint(checkpoint_path)
except Exception as e:
    logger.error(f"Failed to save checkpoint: {e}")
```

## Debugging Commands

### View All Pod Events
```bash
kubectl describe pod <pod-name>
```

### Stream Logs from Both Containers
```bash
# Terminal 1
kubectl logs -f <pod-name> -c node

# Terminal 2
kubectl logs -f <pod-name> -c reward-model
```

### Check Resource Usage
```bash
kubectl top pod <pod-name> --containers
```

### Get Pod YAML
```bash
kubectl get pod <pod-name> -o yaml
```

### Execute Commands in Container
```bash
# Student container
kubectl exec -it <pod-name> -c node -- /bin/bash

# Reward model container
kubectl exec -it <pod-name> -c reward-model -- /bin/bash
```

### Port Forward for Testing
```bash
kubectl port-forward <pod-name> 8080:8080
curl http://localhost:8080/health
```

## Performance Profiling

### Profile Training Loop

Add profiling to `train.py`:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... training code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### GPU Profiling

```bash
kubectl exec <pod-name> -c node -- nvidia-smi dmon -s u -c 100
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# ... training code ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

## Getting Help

If you're still stuck:

1. Collect logs:
```bash
kubectl logs <pod-name> -c node > student.log
kubectl logs <pod-name> -c reward-model > reward.log
kubectl describe pod <pod-name> > pod-describe.txt
```

2. Check TrainJob status:
```bash
kubectl get trainjob <job-name> -o yaml
```

3. Review events:
```bash
kubectl get events --sort-by='.lastTimestamp'
```

4. Open an issue with:
   - Pod logs
   - Pod description
   - TrainJob YAML
   - Error messages
   - Steps to reproduce
