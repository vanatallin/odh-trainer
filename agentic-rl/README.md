# Agentic Reinforcement Learning with Kubeflow Trainer v2

This example demonstrates how to use the Kubeflow Trainer v2 API to train an agentic language model using Reinforcement Learning from Human Feedback (RLHF) with a sidecar reward model.

> **🚀 Quick Start**: Open [openshift-ai-workbench.ipynb](openshift-ai-workbench.ipynb) in your OpenShift AI Workbench!
>
> **What You'll Do:**
> - Submit RL training jobs to Kubernetes using Kubeflow Trainer v2
> - Monitor PPO training progress with reward metrics
> - Access fine-tuned models and test them
>
> **Where's the RL Code?** The PPO algorithm is in [student/train.py](student/train.py) - it runs in GPU containers on Kubernetes. The notebook orchestrates training jobs and monitors results.
>
> **Note**: This example uses the official Kubeflow Training SDK (`TrainerClient` and `CustomTrainer`). See [Kubeflow Trainer Getting Started](https://www.kubeflow.org/docs/components/trainer/getting-started/).

## Overview

This example implements agentic fine-tuning of a small language model using:
- **Student Model**: TinyLlama-1.1B (1.1B parameters) being fine-tuned via Proximal Policy Optimization (PPO)
- **Reward Model (Sidecar)**: A model that scores the quality of generated text based on helpfulness, harmlessness, and honesty
- **Environment**: Text generation tasks where the agent learns to produce better responses

### Why TinyLlama?
- **Small & Efficient**: 1.1B parameters, fits in 8GB GPU memory
- **Fast Training**: Faster iterations for RL experiments
- **Quality**: Built on Llama 2 architecture, delivers good performance
- **RL-Friendly**: Smaller model size enables more stable RL training

## Getting Started

### For OpenShift AI Workbench Users (Recommended)

1. **Upload the notebook**: Upload [openshift-ai-workbench.ipynb](openshift-ai-workbench.ipynb) to your OpenShift AI workbench
2. **Build container images**: Build and push student-agent and reward-model containers
3. **Run the notebook**: Submit training jobs and monitor the PPO training progress
4. **Test the model**: Access checkpoints and test your fine-tuned model
5. **See troubleshooting**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter issues

### For Command-Line Users

You can also deploy using:

1. **Python SDK** - See [sdk/README.md](sdk/README.md) for programmatic usage
2. **YAML Manifests** - See [manifests/](manifests/) for kubectl/oc approach

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    TrainJob Pod                      │
│                                                      │
│  ┌──────────────────┐      ┌──────────────────┐   │
│  │  Student Agent   │─────▶│  Reward Model    │   │
│  │  (RL Training)   │ REST │   (Sidecar)      │   │
│  │                  │◀─────│  HTTP Server     │   │
│  └──────────────────┘      └──────────────────┘   │
│         │                          │               │
│         │                          │               │
│    Shared Volume              Port 8080            │
│   (checkpoints)              (localhost)           │
└─────────────────────────────────────────────────────┘
```

### Communication Flow

1. **Student generates text**: The RL agent generates responses to prompts
2. **Student queries reward model**: POST request to `http://localhost:8080/score` with the prompt and response
3. **Reward model scores**: Returns a reward score (0-1) based on quality
4. **Student updates policy**: Uses PPO to update the model based on rewards
5. **Checkpoint sharing**: Both containers can access shared volume for model checkpoints

## Components

### 1. Student Training Container (`student/`)
- Implements PPO-based RL training loop
- Loads a pre-trained language model (GPT-2)
- Generates text responses to prompts
- Queries the reward model sidecar for scoring
- Updates the policy using PPO

### 2. Reward Model Sidecar (`reward-model/`)
- Exposes a REST API on port 8080
- Loads a pre-trained reward model
- Scores text based on quality criteria
- Returns reward scores to the student

### 3. Kubernetes Manifests (`manifests/`)
- `training-runtime.yaml`: TrainingRuntime with sidecar configuration
- `train-job.yaml`: Example TrainJob for running the training
- `cluster-training-runtime.yaml`: Cluster-scoped runtime (optional)

## Design Patterns

### Why Sidecar Pattern?

The sidecar pattern is ideal for agentic RL fine-tuning because:
- **Low Latency**: Localhost communication minimizes reward query overhead
- **Tight Coupling**: Reward model lifecycle matches training lifecycle
- **Resource Efficiency**: Containers share the same node and can share GPU memory
- **Simplicity**: Single pod deployment, easier debugging

### Alternative: Separate Service Deployment

For production scenarios where you need:
- Multiple training jobs sharing one reward model
- Independent scaling
- Persistent reward model service

See `docs/alternative-patterns.md` for how to deploy the reward model as a separate Kubernetes Service.

## Prerequisites

- Kubernetes cluster with Kubeflow Trainer v2 installed
- GPU nodes (recommended: 1 GPU per training pod)
- kubectl configured to access the cluster
- Container registry for pushing custom images

## Python SDK Quick Start

### Prerequisites

```bash
cd sdk
pip install -r requirements.txt
```

### Deploy with Python SDK

**Option 1: Using Official Kubeflow Trainer API (Recommended)**

```python
from trainjob_client import AgenticRLTrainingClient

# Create client
client = AgenticRLTrainingClient(
    namespace="default",
    student_image="docker.io/myuser/student-agent:latest",
    reward_model_image="docker.io/myuser/reward-model:latest"
)

# Create and deploy train job
train_job = client.create_train_job(
    name="my-rl-training",
    runtime_ref="agentic-rl-pytorch",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs=5,
    batch_size=8,
    learning_rate=2e-5,
)
client.deploy_with_kubectl(train_job)
```

**Option 2: Using Helper SDK**

```python
from trainjob_sdk import AgenticRLTrainingJob, deploy_with_kubectl

# Create job configuration
job = AgenticRLTrainingJob(
    name="my-rl-training",
    namespace="default",
    student_image="docker.io/myuser/student-agent:latest",
    reward_model_image="docker.io/myuser/reward-model:latest"
)

# Create and deploy runtime
runtime = job.create_training_runtime("agentic-rl-pytorch")
deploy_with_kubectl(runtime)

# Create and deploy train job
train_job = job.create_train_job(
    runtime_ref="agentic-rl-pytorch",
    num_epochs=5,
    batch_size=8,
    learning_rate=2e-5,
)
deploy_with_kubectl(train_job)
```

### Or use the command line:

```bash
# Using official API client (recommended)
python sdk/trainjob_client.py \
  --name my-rl-training \
  --student-image docker.io/myuser/student-agent:latest \
  --reward-image docker.io/myuser/reward-model:latest

# Using helper SDK (alternative)
python sdk/trainjob_sdk.py \
  --mode simple \
  --student-image docker.io/myuser/student-agent:latest \
  --reward-image docker.io/myuser/reward-model:latest
```

**See [sdk/README.md](sdk/README.md) for more examples including:**
- Custom prompts via ConfigMap
- Persistent storage
- Weights & Biases integration
- Multi-node distributed training

---

## YAML Quick Start

### Step 1: Build Container Images

```bash
# Build reward model image
cd reward-model
docker build -t <your-registry>/reward-model:latest .
docker push <your-registry>/reward-model:latest

# Build student training image
cd ../student
docker build -t <your-registry>/student-agent:latest .
docker push <your-registry>/student-agent:latest
```

### Step 2: Deploy TrainingRuntime

```bash
kubectl apply -f manifests/training-runtime.yaml
```

### Step 3: Create TrainJob

```bash
# Edit train-job.yaml to update image references
kubectl apply -f manifests/train-job.yaml
```

### Step 4: Monitor Training

```bash
# Watch the job
kubectl get trainjob agentic-rl-training -w

# View logs from student
kubectl logs -f <pod-name> -c student

# View logs from reward model
kubectl logs -f <pod-name> -c reward-model
```

## Configuration

### Student Training Parameters

Edit the TrainJob to configure training:
- `MODEL_NAME`: Language model to use (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `BATCH_SIZE`: Batch size for PPO (default: 4)
- `LEARNING_RATE`: Learning rate (default: 1e-5)
- `PPO_EPOCHS`: PPO optimization epochs (default: 4)
- `REWARD_MODEL_URL`: URL of reward model (default: http://localhost:8080)

**Alternative Models**:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - 1.1B params (default, recommended)
- `microsoft/phi-2` - 2.7B params (requires more GPU memory)
- `mistralai/Mistral-7B-v0.1` - 7B params (requires 16GB+ GPU memory)

### Reward Model Parameters

Configure in the TrainJob:
- `MODEL_NAME`: Hugging Face model for reward scoring
- `DEVICE`: Device to run on (cuda/cpu)

## Training Details

### RL Algorithm: PPO (Proximal Policy Optimization)

This example uses PPO because:
- Stable training for language model fine-tuning
- Prevents catastrophic policy updates
- Well-suited for RLHF scenarios

### Environment

The training environment provides:
- **Prompts**: Questions or instructions requiring helpful responses
- **Actions**: Generated text tokens
- **Rewards**: Scores from the reward model (0-1)
- **Episodes**: Complete prompt-response cycles

### Sample Prompts

The example includes prompts like:
- "How do I make a cake?"
- "Explain quantum physics to a 5-year-old"
- "Write a short poem about nature"

## File Structure

```
agentic-rl/
├── README.md                          # This file
├── QUICKSTART.md                      # 5-minute getting started guide
├── ARCHITECTURE.md                    # Detailed architecture diagrams
├── PROJECT_SUMMARY.md                 # Comprehensive overview
├── CONTRIBUTING.md                    # Contribution guidelines
├── Makefile                           # Build and deployment automation
│
├── sdk/                               # Python SDK (NEW!)
│   ├── README.md                      # SDK documentation
│   ├── trainjob_client.py            # Client SDK (recommended)
│   ├── trainjob_sdk.py               # Full SDK with runtime creation
│   ├── SDK_COMPARISON.md             # SDK comparison guide
│   └── requirements.txt               # SDK dependencies
│
├── manifests/                         # Kubernetes YAML manifests
│   ├── training-runtime.yaml          # TrainingRuntime with sidecar
│   ├── train-job.yaml                 # Example TrainJob
│   └── cluster-training-runtime.yaml  # Cluster-scoped runtime
│
├── student/                           # Student agent code
│   ├── train.py                       # PPO training loop
│   ├── agent.py                       # RL agent implementation
│   ├── environment.py                 # Text generation environment
│   ├── requirements.txt               # Python dependencies
│   └── Dockerfile                     # Student container image
│
├── reward-model/                      # Reward model sidecar
│   ├── server.py                      # REST API server
│   ├── model.py                       # Reward model implementation
│   ├── requirements.txt               # Python dependencies
│   └── Dockerfile                     # Reward model container image
└── docs/
    ├── alternative-patterns.md        # Other deployment patterns
    ├── troubleshooting.md             # Common issues and solutions
    └── scaling.md                     # Scaling considerations
```

## Monitoring and Debugging

### Check Pod Status

```bash
kubectl get pods -l trainjob-name=agentic-rl-training
```

### View Training Metrics

The student container logs metrics every N steps:
- Average reward
- Policy loss
- Value loss
- KL divergence

### Access Reward Model Health

```bash
kubectl port-forward <pod-name> 8080:8080
curl http://localhost:8080/health
```

## Advanced Topics

### Multi-Node Training

To scale to multiple nodes, modify the TrainJob:

```yaml
spec:
  trainer:
    numNodes: 4
    numProcPerNode: 1
    resourcesPerNode:
      requests:
        nvidia.com/gpu: 1
```

**Note**: Each node will have its own reward model sidecar instance.

### Custom Reward Models

Replace the reward model with your own:
1. Update `reward-model/model.py` with your scoring logic
2. Rebuild the container image
3. Update the TrainJob image reference

### Distributed PPO

For large-scale training, see `docs/scaling.md` for distributed PPO setup across multiple nodes.

## References

- [Kubeflow Trainer v2 Proposal](https://github.com/kubeflow/training-operator/blob/master/docs/proposals/2170-kubeflow-trainer-v2/README.md)
- [Training with Human Feedback (OpenAI)](https://openai.com/research/learning-from-human-preferences)
- [PPO Algorithm (Schulman et al.)](https://arxiv.org/abs/1707.06347)
- [TRL - Transformer Reinforcement Learning](https://github.com/huggingface/trl)

## License

Apache 2.0
