# Python SDK for Agentic RL Training

This directory contains Python SDK examples for creating and managing agentic RL training jobs programmatically, replacing YAML-based approaches.

## Overview

Instead of writing YAML manifests, you can now use Python to:
- Create TrainingRuntimes with sidecar configurations
- Define TrainJobs with custom parameters
- Manage volumes, secrets, and ConfigMaps
- Deploy everything programmatically

## SDK Options

This directory contains two SDK implementations:

### 1. `trainjob_client.py` - **Recommended**
- Uses official Kubeflow Training SDK (`TrainerClient` + `CustomTrainer`)
- Clean API focused on job submission
- Type-safe with automatic validation
- Best for production when TrainingRuntime exists

### 2. `trainjob_sdk.py` - Full Setup
- Uses official Kubeflow Training SDK for jobs
- Includes YAML-based TrainingRuntime creation
- Good for initial environment setup
- Creates runtime + submits jobs

### 3. `advanced_sdk_example.py` - ⚠️ Deprecated
- Uses outdated dictionary-based TrainJob creation
- No longer recommended
- Use YAML + kubectl for ConfigMaps/PVCs/Secrets instead

**Need help choosing?** See [SDK_COMPARISON.md](SDK_COMPARISON.md) for a detailed comparison and decision guide.

## Benefits of SDK Approach

1. **Type Safety**: Python type hints catch errors before deployment
2. **Reusability**: Create templates and reuse configurations
3. **Dynamic Configuration**: Programmatically generate configs based on conditions
4. **Version Control**: Track configuration logic in Python code
5. **Testing**: Unit test your deployment configurations

## Installation

```bash
cd sdk
pip install -r requirements.txt
```

## Quick Start

### Basic Single-Node Training

**Recommended: Using Official Kubeflow Training SDK**

```python
from trainjob_client import AgenticRLTrainingClient

# Create client
client = AgenticRLTrainingClient(namespace="default")

# Submit job using upstream SDK (TrainerClient + CustomTrainer)
job_id = client.create_train_job(
    name="my-training-job",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs=5,
    batch_size=8,
    learning_rate=2e-5,
)

# Monitor job
job = client.get_job(job_id)
print(f"Status: {job.status}")

# Stream logs
for logline in client.get_job_logs(job_id, follow=True):
    print(logline)
```

**Alternative: Full Setup with Runtime Creation (trainjob_sdk.py)**

```python
from trainjob_sdk import AgenticRLTrainingSDK

# Create SDK instance
sdk = AgenticRLTrainingSDK(
    namespace="default",
    student_image="docker.io/myuser/student-agent:latest",
    reward_model_image="docker.io/myuser/reward-model:latest"
)

# Create and deploy training runtime (YAML-based)
runtime = sdk.create_training_runtime("agentic-rl-pytorch")
sdk.deploy_runtime(runtime)

# Submit job using upstream SDK (TrainerClient + CustomTrainer)
job_id = sdk.create_train_job(
    name="my-training-job",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs=5,
    batch_size=8,
    learning_rate=2e-5,
)

# Monitor
for logline in sdk.get_job_logs(job_id, follow=True):
    print(logline)
```

### Command Line Usage

**Using trainjob_client.py (Recommended)**

```bash
# Deploy a training job
python trainjob_client.py \
  --name my-training-job \
  --namespace default \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 2e-5

# Follow logs
python trainjob_client.py \
  --name my-training-job \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 5 \
  --follow-logs
```

**Using trainjob_sdk.py (Full Setup)**

```bash
# Create runtime and simple single-node job
python trainjob_sdk.py \
  --mode simple \
  --create-runtime \
  --student-image docker.io/myuser/student-agent:latest \
  --reward-image docker.io/myuser/reward-model:latest

# Distributed multi-node job (runtime already exists)
python trainjob_sdk.py \
  --mode distributed \
  --student-image docker.io/myuser/student-agent:latest \
  --reward-image docker.io/myuser/reward-model:latest \
  --follow-logs
```

## Examples

### Custom Prompts with ConfigMap

Train with domain-specific prompts using ConfigMap:

```bash
# Create ConfigMap from file
cat > prompts.txt <<EOF
Explain the benefits of renewable energy.
How does machine learning differ from traditional programming?
Describe the water cycle in three sentences.
EOF

kubectl create configmap custom-prompts --from-file=prompts.txt

# Submit job that references the ConfigMap
# (Note: Requires modifying TrainingRuntime to mount ConfigMap)
python trainjob_client.py --name custom-prompts-job --epochs 5
```

### Persistent Storage with PVC

Use PersistentVolumeClaim for checkpoints:

```bash
# Create PVC
cat > pvc.yaml <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-checkpoints
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
EOF

kubectl apply -f pvc.yaml

# Submit job (requires TrainingRuntime with PVC mount)
python trainjob_client.py --name persistent-job --epochs 10
```

### Weights & Biases Integration

Integrate with W&B for experiment tracking:

```bash
# Create secret for W&B API key
kubectl create secret generic wandb-secret \
  --from-literal=api-key=YOUR_WANDB_API_KEY

# Submit job (requires TrainingRuntime with W&B env vars)
python trainjob_client.py --name wandb-job --epochs 5
```

## Advanced Usage

### Multi-Node Distributed Training

```python
from trainjob_sdk import AgenticRLTrainingSDK

sdk = AgenticRLTrainingSDK(
    namespace="ml-team",
    student_image="myregistry/student:v2",
    reward_model_image="myregistry/reward:v2"
)

# Create runtime for multi-node setup
runtime = sdk.create_training_runtime("agentic-rl-pytorch-multi")
sdk.deploy_runtime(runtime)

# Submit multi-node job
job_id = sdk.create_train_job(
    name="large-scale-training",
    model_name="gpt2-medium",
    num_epochs=20,
    batch_size=16,
    learning_rate=5e-5,
    num_nodes=2,               # Multi-node
    gpu_per_node=4,            # 4 GPUs per node
    memory_per_node="32Gi",
    cpu_per_node=16,
)

# Monitor
for logline in sdk.get_job_logs(job_id, follow=True):
    print(logline)
```

### Cluster-Scoped Runtime

```python
from trainjob_sdk import AgenticRLTrainingSDK

sdk = AgenticRLTrainingSDK(
    student_image="myregistry/student:latest",
    reward_model_image="myregistry/reward:latest"
)

# Create ClusterTrainingRuntime (cluster-wide, no namespace)
runtime = sdk.create_training_runtime(
    runtime_name="agentic-rl-pytorch",
    cluster_scoped=True  # Creates ClusterTrainingRuntime
)
sdk.deploy_runtime(runtime)
```

## Integration with Existing Workflows

### Jupyter Notebook

```python
# In a Jupyter notebook
from trainjob_client import AgenticRLTrainingClient

# Create client
client = AgenticRLTrainingClient(namespace="default")

# Submit job
job_id = client.create_train_job(
    name="notebook-job",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs=5,
)

# Monitor
job = client.get_job(job_id)
print(f"Status: {job.status}")

# View logs
for logline in client.get_job_logs(job_id, follow=True):
    print(logline)
```

### CI/CD Pipeline

```python
# In GitHub Actions / GitLab CI
import os
from trainjob_client import AgenticRLTrainingClient

# Configure based on environment
if os.environ['CI_COMMIT_BRANCH'] == 'main':
    namespace = "production"
    num_epochs = 50
else:
    namespace = "staging"
    num_epochs = 3

# Submit job
client = AgenticRLTrainingClient(namespace=namespace)
job_id = client.create_train_job(
    name=f"training-{os.environ['CI_COMMIT_SHA'][:8]}",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs=num_epochs,
)

# Wait for completion
job = client.get_job(job_id)
print(f"Job {job_id} status: {job.status}")
```

### Python Script with Args

```python
import argparse
from trainjob_client import AgenticRLTrainingClient

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
args = parser.parse_args()

client = AgenticRLTrainingClient()
job_id = client.create_train_job(
    name=args.name,
    model_name=args.model,
    num_epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.lr,
)
print(f"Created job: {job_id}")
```

## API Reference

### `AgenticRLTrainingClient` (trainjob_client.py) - **Recommended**

Client using official Kubeflow Training SDK.

**Constructor:**
```python
AgenticRLTrainingClient(
    namespace: str = "default"
)
```

**Methods:**

#### `create_train_job()`
```python
create_train_job(
    name: str,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    ppo_epochs: int = 4,
    max_response_length: int = 128,
    num_nodes: int = 1,
    gpu_per_node: int = 1,
    memory_per_node: str = "8Gi",
    cpu_per_node: int = 4,
) -> str
```
Submits a TrainJob using `TrainerClient` and `CustomTrainer`. Returns the job ID (name).

#### `get_job()`
```python
get_job(name: str)
```
Get job details.

#### `get_job_logs()`
```python
get_job_logs(name: str, follow: bool = False)
```
Get job logs. If `follow=True`, streams logs in real-time.

#### `list_jobs()`
```python
list_jobs()
```
List all training jobs in the namespace.

#### `delete_job()`
```python
delete_job(name: str)
```
Delete a training job.

---

### `AgenticRLTrainingSDK` (trainjob_sdk.py) - Full Setup

SDK using official Kubeflow Training SDK with runtime creation capabilities.

**Constructor:**
```python
AgenticRLTrainingSDK(
    namespace: str = "default",
    student_image: str = "your-registry/student-agent:latest",
    reward_model_image: str = "your-registry/reward-model:latest"
)
```

**Methods:**

#### `create_training_runtime()`
```python
create_training_runtime(
    runtime_name: str = "agentic-rl-pytorch",
    cluster_scoped: bool = False
) -> Dict
```
Creates a TrainingRuntime or ClusterTrainingRuntime with sidecar configuration (YAML dict).

#### `deploy_runtime()`
```python
deploy_runtime(
    runtime_spec: Dict,
    dry_run: bool = False
)
```
Deploys a TrainingRuntime using kubectl.

#### `create_train_job()`
```python
create_train_job(
    name: str,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    ppo_epochs: int = 4,
    max_response_length: int = 128,
    num_nodes: int = 1,
    gpu_per_node: int = 1,
    memory_per_node: str = "8Gi",
    cpu_per_node: int = 4,
) -> str
```
Submits a TrainJob using `TrainerClient` and `CustomTrainer`. Returns the job ID (name).

#### `get_job()`, `get_job_logs()`, `list_jobs()`, `delete_job()`
Same as `AgenticRLTrainingClient` methods.


## Comparison: YAML vs SDK

### YAML Approach

```yaml
# training-runtime.yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainingRuntime
metadata:
  name: agentic-rl-pytorch
spec:
  mlPolicy:
    numNodes: 1
    torch:
      numProcPerNode: auto
  # ... 100+ lines ...

---
# train-job.yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: agentic-rl-training
spec:
  runtimeRef:
    name: agentic-rl-pytorch
  # ... 50+ lines ...
```

```bash
kubectl apply -f training-runtime.yaml
kubectl apply -f train-job.yaml
```

### SDK Approach

```python
from trainjob_client import AgenticRLTrainingClient

client = AgenticRLTrainingClient(namespace="default")

job_id = client.create_train_job(
    name="agentic-rl-training",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs=5,
    batch_size=8,
)

# Monitor
for logline in client.get_job_logs(job_id, follow=True):
    print(logline)
```

**Advantages:**
- Uses official Kubeflow Training SDK
- Type-safe with IDE autocomplete
- Reusable across projects
- Programmatically configurable
- Easier to test and validate
- Version control friendly
- Built-in job monitoring

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you're in the sdk directory
cd sdk
pip install -r requirements.txt
```

### Kubectl Not Found

Install kubectl:
```bash
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
```

### Permission Denied

Make sure scripts are executable:
```bash
chmod +x trainjob_sdk.py advanced_sdk_example.py
```

### Validation Errors

Use dry-run to validate:
```bash
python trainjob_sdk.py --dry-run ...
```

## Official Kubeflow Training SDK Reference

This implementation uses the official Kubeflow Training SDK. For more details, see [Kubeflow Trainer Getting Started](https://www.kubeflow.org/docs/components/trainer/getting-started/).

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing SDK examples.

## License

Apache 2.0
