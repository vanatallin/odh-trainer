# Machine Pool Creation Guide for ROSA

This guide shows how to create GPU and CPU machine pools in Red Hat OpenShift Service on AWS (ROSA).

## Prerequisites

- ROSA cluster running
- `rosa` CLI installed and configured
- AWS account with appropriate permissions

## GPU Machine Pool

Create a machine pool with GPU nodes for training workloads:

```bash
rosa create machinepool \
  --cluster=my-cluster \
  --name=gpu-pool \
  --replicas=2 \
  --instance-type=g5.2xlarge \
  --labels=workload=gpu-training,nvidia.com/gpu.present=true
```

### GPU Instance Types

| Instance Type | GPUs | GPU Model | vCPUs | RAM | Use Case |
|--------------|------|-----------|-------|-----|----------|
| `g5.xlarge` | 1 | NVIDIA A10G | 4 | 16 GB | Small models |
| `g5.2xlarge` | 1 | NVIDIA A10G | 8 | 32 GB | **Recommended** |
| `g5.4xlarge` | 1 | NVIDIA A10G | 16 | 64 GB | Large models |
| `g5.12xlarge` | 4 | NVIDIA A10G | 48 | 192 GB | Multi-GPU |
| `p3.2xlarge` | 1 | NVIDIA V100 | 8 | 61 GB | High memory |
| `p3.8xlarge` | 4 | NVIDIA V100 | 32 | 244 GB | Multi-GPU V100 |

**Cost Note:** GPU instances are expensive! Use autoscaling to save costs.

## CPU Machine Pool (High Performance)

For data preprocessing and CPU-intensive workloads:

```bash
rosa create machinepool \
  --cluster=my-cluster \
  --name=cpu-pool \
  --replicas=3 \
  --instance-type=c5.4xlarge \
  --labels=workload=cpu-intensive
```

### CPU Instance Types

| Instance Type | vCPUs | RAM | Use Case |
|--------------|-------|-----|----------|
| `c5.2xlarge` | 8 | 16 GB | Light preprocessing |
| `c5.4xlarge` | 16 | 32 GB | **Recommended** |
| `c5.9xlarge` | 36 | 72 GB | Heavy preprocessing |
| `c5.12xlarge` | 48 | 96 GB | Massive datasets |

## Autoscaling (Cost Optimization)

Scale GPU pool to zero when not in use:

```bash
rosa create machinepool \
  --cluster=my-cluster \
  --name=gpu-pool \
  --enable-autoscaling \
  --min-replicas=0 \
  --max-replicas=5 \
  --instance-type=g5.2xlarge \
  --labels=workload=gpu-training,nvidia.com/gpu.present=true
```

**Benefits:**
- Scale to 0 replicas when no training jobs running → **$0 cost**
- Auto-scale up when jobs submitted
- Auto-scale down when jobs complete

## Verify Machine Pool

```bash
# List all machine pools
rosa list machinepools --cluster=my-cluster

# Check specific pool
rosa describe machinepool --cluster=my-cluster --machinepool=gpu-pool

# Verify nodes in OpenShift
oc get nodes -l nvidia.com/gpu.present=true
oc get nodes -l workload=gpu-training
```

## Update Machine Pool

### Scale Up/Down

```bash
# Manual scaling
rosa update machinepool \
  --cluster=my-cluster \
  --machinepool=gpu-pool \
  --replicas=5

# Enable autoscaling
rosa update machinepool \
  --cluster=my-cluster \
  --machinepool=gpu-pool \
  --enable-autoscaling \
  --min-replicas=0 \
  --max-replicas=10
```

### Change Instance Type

You cannot change instance type. Instead:

1. Create new machine pool with desired instance type
2. Delete old machine pool

```bash
# Create new pool
rosa create machinepool \
  --cluster=my-cluster \
  --name=gpu-pool-v2 \
  --instance-type=g4dn.xlarge \
  --replicas=2 \
  --labels=workload=gpu-training,nvidia.com/gpu.present=true

# Wait for nodes to be ready
oc get nodes -l workload=gpu-training -w

# Delete old pool
rosa delete machinepool --cluster=my-cluster --machinepool=gpu-pool
```

## Delete Machine Pool

```bash
rosa delete machinepool --cluster=umangani-dev --machinepool=gpu-pool
```

**Warning:** This will terminate all nodes and drain running pods!

## Using Machine Pools for Training

Once your GPU machine pool is created, TrainJobs will automatically schedule on available GPU nodes:

```python
from sdk.trainjob_sdk import AgenticRLTrainingSDK

sdk = AgenticRLTrainingSDK(
    namespace='my-namespace',
    student_image='quay.io/user/student:latest',
    reward_model_image='quay.io/user/reward:latest'
)

# Request GPUs - will schedule on gpu-pool
job_id = sdk.create_train_job(
    name='my-training-job',
    gpu_per_node=1,  # Schedules on GPU nodes
    num_nodes=2,
    memory_per_node='16Gi',
    cpu_per_node=8
)
```

## Monitoring Costs

GPU instances can be expensive. Monitor usage:

```bash
# Check current replicas
rosa list machinepools --cluster=my-cluster

# Check node utilization
oc adm top nodes

# Check GPU allocation
oc describe nodes -l nvidia.com/gpu.present=true | grep -A 5 "Allocated resources"
```

**Cost-saving tips:**
1. Use autoscaling with `min-replicas=0`
2. Delete idle machine pools
3. Use spot instances for fault-tolerant workloads
4. Schedule training jobs during off-peak hours

## Example: Complete Setup

```bash
# 1. Create GPU pool with autoscaling
rosa create machinepool \
  --cluster=ml-cluster \
  --name=gpu-training \
  --enable-autoscaling \
  --min-replicas=0 \
  --max-replicas=4 \
  --instance-type=g5.2xlarge \
  --labels=workload=gpu-training,nvidia.com/gpu.present=true

# 2. Create CPU pool for preprocessing
rosa create machinepool \
  --cluster=ml-cluster \
  --name=cpu-preprocessing \
  --replicas=2 \
  --instance-type=c5.4xlarge \
  --labels=workload=cpu-intensive

# 3. Verify
rosa list machinepools --cluster=ml-cluster

# 4. Check nodes (may take 5-10 minutes)
watch oc get nodes

# 5. Verify GPU nodes have capacity
oc get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns=NAME:.metadata.name,GPUS:.status.capacity.'nvidia\.com/gpu'
```

## Troubleshooting

### Nodes Not Appearing

Wait 5-10 minutes for AWS to provision EC2 instances.

```bash
# Check machine pool status
rosa describe machinepool --cluster=my-cluster --machinepool=gpu-pool

# Check for errors
rosa logs cluster --cluster=my-cluster --watch
```

### GPU Not Available

Check GPU device plugin is running:

```bash
oc get daemonset -n nvidia-gpu-operator
oc get pods -n nvidia-gpu-operator
```

If not installed, OpenShift should auto-install when GPU nodes appear.

### Cost Concerns

**Estimated costs** (us-east-1, on-demand, Jan 2026):
- g5.xlarge: ~$1.00/hour
- g5.2xlarge: ~$1.21/hour
- g5.4xlarge: ~$1.62/hour
- p3.2xlarge: ~$3.06/hour

**Always use autoscaling** to minimize costs!

## References

- [ROSA Machine Pools Documentation](https://docs.openshift.com/rosa/rosa_cluster_admin/rosa_nodes/rosa-managing-worker-nodes.html)
- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/openshift/contents.html)
