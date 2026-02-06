# OpenShift AI Workbench Quickstart Guide

Get started with agentic RL training in OpenShift AI Workbench in 15 minutes!

## Overview

This guide shows you how to:
1. Set up your OpenShift AI Workbench environment
2. Build and push container images
3. Run the interactive notebook to train an LLM with reinforcement learning
4. Monitor training progress and access your fine-tuned model

## Prerequisites

### OpenShift AI Setup
- ✅ OpenShift AI installed on your cluster
- ✅ Data Science Project created (e.g., `my-ai-project`)
- ✅ Workbench created with:
  - **Image**: Standard Data Science notebook (includes Python, Jupyter, oc CLI)
  - **Size**: Small or Medium (notebook orchestrates, doesn't train)
  - **Storage**: At least 20GB for code and model checkpoints

### GPU Resources
- ✅ GPU quota allocated to your namespace/project
- ✅ At least 1 GPU node available in your cluster
- ✅ Node tolerations configured (if GPU nodes are tainted)

### Additional Requirements
- ✅ Container registry access (Quay.io, Docker Hub, or internal registry)
- ✅ Registry credentials configured in OpenShift
- ✅ Kubeflow Training Operator installed (see [Installation](#installing-kubeflow-training-operator))

## Step 1: Access Your Workbench

1. Log into OpenShift AI Dashboard
2. Navigate to your Data Science Project
3. Click **Open** on your workbench
4. Wait for JupyterLab to load

## Step 2: Clone the Repository

In the JupyterLab terminal:

```bash
cd ~
git clone https://github.com/your-org/odh-trainer.git
cd odh-trainer/agentic-rl
```

Or if you have the code already, upload the `agentic-rl` folder to your workbench.

## Step 3: Set Up Container Registry

### Option A: Using Quay.io

1. Create a Quay.io account at https://quay.io
2. Create a new repository: `student-agent`
3. Create another repository: `reward-model`
4. Get your credentials:
   ```bash
   # Login to Quay.io
   podman login quay.io
   # Enter your username and password
   ```

### Option B: Using OpenShift Internal Registry

```bash
# Get the internal registry URL
REGISTRY=$(oc get route default-route -n openshift-image-registry -o jsonpath='{.spec.host}')

# Login to internal registry
podman login -u $(oc whoami) -p $(oc whoami -t) $REGISTRY
```

### Set Registry Environment Variable

```bash
# For Quay.io
export REGISTRY=quay.io/your-username

# For OpenShift internal registry
export REGISTRY=$(oc get route default-route -n openshift-image-registry -o jsonpath='{.spec.host}')/$(oc project -q)

# Verify
echo $REGISTRY
```

## Step 4: Build Container Images

Build both the student training agent and reward model containers:

```bash
# Build student agent image
make build-student REGISTRY=$REGISTRY

# Build reward model image
make build-reward REGISTRY=$REGISTRY
```

**Expected output:**
```
Building student-agent...
Successfully tagged quay.io/your-username/student-agent:latest

Building reward-model...
Successfully tagged quay.io/your-username/reward-model:latest
```

**Alternative: Build both at once**
```bash
make build REGISTRY=$REGISTRY
```

## Step 5: Push Images to Registry

```bash
# Push both images
make push REGISTRY=$REGISTRY
```

**Verify images are pushed:**
```bash
# For Quay.io - check https://quay.io/repository/your-username/student-agent

# For OpenShift internal registry
oc get imagestream -n $(oc project -q)
```

## Step 6: Get Your Namespace

```bash
# Get current project/namespace
export NAMESPACE=$(oc project -q)
echo "Your namespace: $NAMESPACE"

# Example output: my-ai-project
```

## Step 7: Open the Notebook

1. In JupyterLab file browser, navigate to `agentic-rl/`
2. Double-click `openshift-ai-workbench.ipynb`
3. Wait for the notebook to load

## Step 8: Configure the Notebook

In **Cell 4** of the notebook, update these variables:

```python
# Configuration
NAMESPACE = "my-ai-project"  # ← Your Data Science Project namespace
STUDENT_IMAGE = "quay.io/your-username/student-agent:latest"  # ← Your student image
REWARD_IMAGE = "quay.io/your-username/reward-model:latest"    # ← Your reward image

# Machine Pool Configuration
WORKLOAD_TYPE = "gpu-power"  # Options: "gpu-power", "cpu-power", "workers"
```

### Understanding Machine Pools

The notebook supports targeting different node pools in your cluster:

| Pool Type | When to Use | Configuration |
|-----------|-------------|---------------|
| **gpu-power** | GPU training (recommended) | Requires GPU nodes with `workload=gpu-training` label |
| **cpu-power** | CPU-intensive training | Requires CPU nodes with `workload=cpu-intensive` label |
| **workers** | General workloads | Standard worker nodes |

**Check your node labels:**
```bash
oc get nodes --show-labels | grep workload
```

**If your GPU nodes have different labels**, update the `GPU_POOL_CONFIG` in Cell 4:
```python
GPU_POOL_CONFIG = {
    "tolerations": [
        {"key": "nvidia.com/gpu", "operator": "Equal", "value": "present", "effect": "NoSchedule"}
    ],
    "nodeSelector": {
        "nvidia.com/gpu.present": "true"
    }
}
```

## Step 9: Run the Notebook

Execute cells in order (or run **Run > Run All Cells**):

### Key Sections:

**Cells 1-6: Setup**
- Install dependencies
- Configure SDK
- Initialize client

**Cells 7-9: Deploy TrainingRuntime**
- Creates the pod template with sidecar pattern
- Deploys to your namespace
- Verifies creation

**Cells 10-12: Submit Training Job**
- Configures training parameters:
  - Model: TinyLlama-1.1B-Chat (1.1B parameters)
  - Epochs: 3
  - Learning Rate: 1e-5
  - PPO Epochs: 4
- Submits job to Kubernetes
- Job runs in GPU pods, not in notebook

**Cells 13-18: Monitor Training**
- Check job and pod status
- View logs in real-time
- Verify GPU node scheduling

**Cells 19-25: View Metrics**
- Extract training metrics from logs
- Plot reward progression
- See training statistics

**Cells 26-29: Access Checkpoints**
- Copy trained model from pod to notebook storage
- Save checkpoints locally

**Cells 30-33: Test Model**
- Load fine-tuned model in notebook
- Run inference on test prompts
- Compare outputs

## Step 10: Monitor Training Progress

### Check Job Status

```bash
oc get trainjob -n $NAMESPACE
```

Expected output:
```
NAME                            STATE      AGE
agentic-rl-20260105-143000     Running    5m
```

### Check Pod Status

```bash
oc get pods -n $NAMESPACE -l trainjob-name=agentic-rl-20260105-143000
```

Expected output:
```
NAME                                     READY   STATUS    RESTARTS   AGE
agentic-rl-20260105-143000-node-0-0-qx4  2/2     Running   0          5m
```

**Note:** `2/2` means both containers (student + reward model) are running.

### View Training Logs

**Option A: In notebook (Cell 20)**
```bash
!oc logs -f -l trainjob-name={job_id} -c node -n {NAMESPACE}
```

**Option B: In terminal**
```bash
POD=$(oc get pods -n $NAMESPACE -l trainjob-name=agentic-rl-20260105-143000 -o jsonpath='{.items[0].metadata.name}')
oc logs -f $POD -c node -n $NAMESPACE
```

**Expected log output:**
```
Epoch 1/3, Step 1/100
Generated: "How to make chocolate chip cookies? Here's a simple recipe..."
Reward: 0.85
Avg Reward: 0.85

Epoch 1/3, Step 2/100
Generated: "The water cycle involves evaporation, condensation..."
Reward: 0.92
Avg Reward: 0.885
```

### Check Reward Model Logs

```bash
oc logs -f $POD -c reward-model -n $NAMESPACE
```

Expected output:
```
Reward model server started on http://0.0.0.0:8080
Loaded model: reward-model-v1
Received scoring request: prompt="How to make chocolate chip cookies?"
Scored response: reward=0.85
```

## Step 11: Verify GPU Scheduling

The notebook includes a detailed verification cell (Cell 14) that checks:

- ✅ Pod is scheduled on correct machine pool
- ✅ Node has expected workload label
- ✅ GPU labels are present (for gpu-power pool)
- ✅ Tolerations are correctly applied
- ✅ Instance type is shown

**Manual verification:**
```bash
# Get pod name
POD=$(oc get pods -n $NAMESPACE -l trainjob-name=agentic-rl-20260105-143000 -o jsonpath='{.items[0].metadata.name}')

# Check which node it's on
NODE=$(oc get pod $POD -n $NAMESPACE -o jsonpath='{.spec.nodeName}')
echo "Pod is on node: $NODE"

# Check node labels
oc get node $NODE --show-labels | grep -E '(workload|nvidia)'
```

Expected output:
```
Pod is on node: gpu-worker-1
workload=gpu-training,nvidia.com/gpu.present=true
```

## Step 12: Access Your Fine-Tuned Model

Once training completes (check with `oc get trainjob`), copy checkpoints:

**In notebook (Cell 29):**
```bash
checkpoint_dir = f"./checkpoints/{job_id}"
!mkdir -p {checkpoint_dir}
!oc cp {pod_name}:/checkpoints/final {checkpoint_dir}/final -c node -n {NAMESPACE}
```

**Checkpoints location:** `~/odh-trainer/agentic-rl/checkpoints/{job_id}/final/`

**Test the model (Cells 32-33):**
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(f"./checkpoints/{job_id}/final")
tokenizer = AutoTokenizer.from_pretrained(f"./checkpoints/{job_id}/final")

# Generate text
prompt = "How do I make chocolate chip cookies?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150)
print(tokenizer.decode(outputs[0]))
```

## Troubleshooting

### Issue: Pod stuck in `Pending` state

**Check node availability:**
```bash
oc get nodes -l workload=gpu-training
```

**If no GPU nodes:**
- Verify GPU machine pool exists
- Check node labels: `oc describe node <node-name>`
- See [MACHINE_POOLS_GUIDE.md](MACHINE_POOLS_GUIDE.md)

### Issue: Pod stuck in `ImagePullBackOff`

**Check image URL:**
```bash
oc describe pod $POD -n $NAMESPACE | grep -A5 "Events:"
```

**Solutions:**
- Verify image was pushed: `podman images | grep student-agent`
- Check registry credentials: `oc get secret -n $NAMESPACE`
- Make Quay.io repository public or create pull secret

### Issue: Out of GPU memory

**Reduce model size or batch size** in notebook Cell 11:
```python
BATCH_SIZE = 2  # Reduce from 4
MODEL_NAME = "gpt2"  # Use smaller model instead of TinyLlama
```

### Issue: Training job fails with "Connection refused" to reward model

**Check reward model container logs:**
```bash
oc logs $POD -c reward-model -n $NAMESPACE
```

**Verify both containers are running:**
```bash
oc get pod $POD -n $NAMESPACE -o jsonpath='{.status.containerStatuses[*].name}'
```

Expected: `node reward-model`

### Issue: oc command not found in notebook

**Install OpenShift CLI in workbench:**
```bash
curl -LO https://mirror.openshift.com/pub/openshift-v4/clients/ocp/stable/openshift-client-linux.tar.gz
tar -xvf openshift-client-linux.tar.gz
sudo mv oc /usr/local/bin/
oc version
```

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Installing Kubeflow Training Operator

If Kubeflow Training Operator is not installed:

### Option A: RHOAI Deployment (Recommended for OpenShift AI)

```bash
# From the repository root
cd ~/odh-trainer

# Deploy to opendatahub namespace (default for RHOAI)
make deploy-rhoai NAMESPACE=opendatahub

# Verify installation
oc get deployment kubeflow-trainer-controller-manager -n opendatahub
```

### Option B: Helm Chart (Upstream)

```bash
helm repo add kubeflow https://kubeflow.github.io/helm-charts
helm repo update
helm install kubeflow-trainer kubeflow/kubeflow-trainer -n kubeflow --create-namespace
```

### Verify Installation

```bash
# Check CRDs are installed
oc get crd trainjobs.trainer.kubeflow.org
oc get crd trainingruntimes.trainer.kubeflow.org

# Check controller is running
oc get pods -n opendatahub -l app=kubeflow-trainer-controller-manager
```

## Next Steps

### Experiment with Different Models

Update Cell 11 in the notebook:
```bash
MODEL_NAME = "microsoft/phi-2"  # 2.7B parameters
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # 7B parameters (requires more GPU memory)
```

### Customize Training Prompts

Edit `student/environment.py` to add your own training tasks:
```python
TRAINING_PROMPTS = [
    "Explain quantum computing to a 10-year-old",
    "Write a haiku about artificial intelligence",
    # Add your custom prompts here
]
```

Rebuild and push:
```bash
make build-student push-student REGISTRY=$REGISTRY
```

### Run Hyperparameter Sweep

Use Cell 35 in the notebook to run multiple experiments:
```python
experiments = [
    {"learning_rate": 1e-5, "batch_size": 4, "ppo_epochs": 4},
    {"learning_rate": 2e-5, "batch_size": 4, "ppo_epochs": 4},
    {"learning_rate": 1e-5, "batch_size": 8, "ppo_epochs": 8},
]
```

### Scale to Multi-Node Training

See [README.md#Scaling](README.md#scaling) for distributed training across multiple GPUs/nodes.

### Add Persistent Storage

See [docs/persistent-storage.md](docs/persistent-storage.md) for saving checkpoints to PVCs.

## Clean Up

**Delete training job (keeps runtime for reuse):**
```bash
# In notebook Cell 37
!oc delete trainjob {job_id} -n {NAMESPACE}
```

**Delete everything (job + runtime):**
```bash
# Delete all training jobs
oc delete trainjobs --all -n $NAMESPACE

# Delete training runtime
oc delete trainingruntime agentic-rl-pytorch -n $NAMESPACE
```

## Additional Resources

### Documentation
- [Full README](README.md) - Complete project documentation
- [Architecture Guide](ARCHITECTURE.md) - Deep dive into system design
- [SDK Documentation](sdk/README.md) - Python SDK reference
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions

### Kubeflow Resources
- [Kubeflow Trainer Documentation](https://www.kubeflow.org/docs/components/trainer/)
- [Kubeflow SDK Guide](https://github.com/kubeflow/sdk)

### OpenShift AI Resources
- [OpenShift AI Documentation](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/)
- [Data Science Projects Guide](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.13/html/working_on_data_science_projects/)

## Getting Help

- **Issues**: Open an issue at [GitHub Issues](https://github.com/your-org/odh-trainer/issues)
- **Slack**: Join `#kubeflow-trainer` on [Kubeflow Slack](https://www.kubeflow.org/docs/about/community/#kubeflow-slack)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Happy Training! 🚀**

Built with ❤️ using Kubeflow Trainer v2 and OpenShift AI
