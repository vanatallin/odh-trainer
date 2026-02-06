# Architecture Documentation

## System Overview

This document provides detailed architecture diagrams and explanations for the agentic RL training system.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    Kubeflow Trainer v2                      │   │
│  │                  (Training Operator)                        │   │
│  └───────────────────────┬────────────────────────────────────┘   │
│                          │                                          │
│                          │ Watches TrainJob CRD                     │
│                          ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                 TrainingRuntime                             │   │
│  │  (Blueprint for training pods with sidecar)                 │   │
│  └───────────────────────┬────────────────────────────────────┘   │
│                          │                                          │
│                          │ Creates JobSet                           │
│                          ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                      JobSet                                 │   │
│  │  (Manages group of Jobs)                                    │   │
│  └───────────────────────┬────────────────────────────────────┘   │
│                          │                                          │
│                          │ Creates Job(s)                           │
│                          ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                       Job                                   │   │
│  │  (Kubernetes batch Job)                                     │   │
│  └───────────────────────┬────────────────────────────────────┘   │
│                          │                                          │
│                          │ Creates Pod(s)                           │
│                          ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                   Training Pod                              │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐       │   │
│  │  │  Student Container   │  │ Reward Model Sidecar │       │   │
│  │  │  (Main Training)     │  │  (REST API)          │       │   │
│  │  └──────────────────────┘  └──────────────────────┘       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Pod Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         Training Pod                               │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Student Container (node)                        │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────┐          │ │
│  │  │         PPO Agent                             │          │ │
│  │  │  ┌────────────────┐  ┌────────────────┐     │          │ │
│  │  │  │ Policy Network │  │ Value Network  │     │          │ │
│  │  │  │   (GPT-2)      │  │  (Optional)    │     │          │ │
│  │  │  └────────────────┘  └────────────────┘     │          │ │
│  │  │                                              │          │ │
│  │  │  ┌────────────────────────────────────┐    │          │ │
│  │  │  │  Reference Model (Frozen)          │    │          │ │
│  │  │  │  (For KL divergence)               │    │          │ │
│  │  │  └────────────────────────────────────┘    │          │ │
│  │  └──────────────────────────────────────────────┘          │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────┐          │ │
│  │  │         Environment                           │          │ │
│  │  │  - Prompts dataset                           │          │ │
│  │  │  - Episode tracking                          │          │ │
│  │  │  - Statistics                                │          │ │
│  │  └──────────────────────────────────────────────┘          │ │
│  │                                                              │ │
│  │  Network Interface: eth0                                    │ │
│  │  Volumes: /workspace, /checkpoints, /dev/shm               │ │
│  │  GPU: nvidia.com/gpu (shared)                              │ │
│  └──────────────────┬───────────────────────────────────────┬─┘ │
│                     │                                       │   │
│                     │ HTTP POST /score                      │   │
│                     │ localhost:8080                        │   │
│                     │                                       │   │
│  ┌──────────────────▼───────────────────────────────────────▼─┐ │
│  │              Reward Model Container                        │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────┐          │ │
│  │  │         REST API Server                       │          │ │
│  │  │  - POST /score → reward                      │          │ │
│  │  │  - GET  /health → status                     │          │ │
│  │  └──────────────────────────────────────────────┘          │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────┐          │ │
│  │  │    Reward Model                               │          │ │
│  │  │    (DistilBERT or custom)                    │          │ │
│  │  │    - Tokenizer                               │          │ │
│  │  │    - Model (sequence classification)         │          │ │
│  │  └──────────────────────────────────────────────┘          │ │
│  │                                                              │ │
│  │  Listens on: 0.0.0.0:8080                                  │ │
│  │  Volumes: /checkpoints (ro), /dev/shm                      │ │
│  │  GPU: nvidia.com/gpu (shared)                              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                    │
│  Shared Resources:                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Volume: checkpoints (emptyDir)                            │   │
│  │ Volume: dshm (emptyDir, Memory)                          │   │
│  │ GPU: Shared between containers                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

## Training Loop Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Loop                                 │
└─────────────────────────────────────────────────────────────────┘

Start
  │
  ├─► Wait for Reward Model Ready
  │   └─► Poll http://localhost:8080/health
  │
  ├─► Load Model & Environment
  │   ├─► Load GPT-2 (policy)
  │   ├─► Load GPT-2 (reference, frozen)
  │   └─► Load prompts
  │
  └─► For each Epoch:
      │
      └─► For each Batch:
          │
          ├─► 1. Generate Trajectories
          │   │
          │   └─► For each Prompt in Batch:
          │       ├─► Generate response (policy.generate())
          │       │   Output: tokens, log_probs
          │       │
          │       ├─► Query Reward Model
          │       │   │
          │       │   ├─► POST http://localhost:8080/score
          │       │   │   Body: {prompt, response}
          │       │   │
          │       │   └─► Response: {score: 0.85}
          │       │
          │       └─► Store: (prompt, response, log_probs, reward)
          │
          ├─► 2. Compute Advantages
          │   └─► GAE (Generalized Advantage Estimation)
          │       Output: advantages, returns
          │
          ├─► 3. PPO Update (for K epochs)
          │   │
          │   └─► For each Trajectory:
          │       ├─► Forward pass (get new log_probs)
          │       ├─► Compute policy loss (clipped)
          │       ├─► Compute value loss (optional)
          │       ├─► Compute entropy bonus
          │       ├─► Compute KL divergence
          │       ├─► Backward pass
          │       └─► Optimizer step
          │
          └─► 4. Logging
              └─► Log metrics (reward, loss, KL)
      │
      └─► Save Checkpoint
          └─► model.save_pretrained(/checkpoints/epoch-N)

End
```

## Communication Flow

```
Student Container                          Reward Model Container
      │                                           │
      │ 1. Generate response                      │
      │    "How to bake a cake?"                  │
      │    → "Mix flour, eggs..."                 │
      │                                           │
      │ 2. Prepare request                        │
      │    {                                      │
      │      "prompt": "How to bake...",         │
      │      "response": "Mix flour..."          │
      │    }                                      │
      │                                           │
      │ 3. HTTP POST                              │
      │    http://localhost:8080/score            │
      ├──────────────────────────────────────────▶│
      │                                           │
      │                                      4. Tokenize
      │                                         prompt + response
      │                                           │
      │                                      5. Forward pass
      │                                         through model
      │                                           │
      │                                      6. Compute score
      │                                         sigmoid(logits)
      │                                           │
      │ 7. HTTP Response                          │
      │    {                                      │
      │      "score": 0.85,                      │
      │      "prompt": "How to...",              │
      │      "response_length": 234              │
      │    }                                      │
      │◀──────────────────────────────────────────┤
      │                                           │
      │ 8. Use score as reward                    │
      │    r = 0.85                               │
      │                                           │
      │ 9. Update policy                          │
      │    loss = ppo_loss(reward=r)             │
      │    loss.backward()                        │
      │                                           │
      ▼                                           ▼
```

## Multi-Node Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Distributed Training                          │
│                    (4 Nodes, 8 GPUs total)                       │
└─────────────────────────────────────────────────────────────────┘

Node 0 (Rank 0, 1)              MASTER_ADDR=node-0
┌──────────────────────┐
│  Student + Reward    │        All-Reduce Gradients
│  GPU 0, GPU 1        │        (NCCL Backend)
└──────────┬───────────┘               │
           │                           │
           ├───────────────────────────┼────────────────┐
           │                           │                │
Node 1 (Rank 2, 3)          Node 2 (Rank 4, 5)    Node 3 (Rank 6, 7)
┌──────────────────────┐   ┌──────────────────────┐  ┌──────────────────────┐
│  Student + Reward    │   │  Student + Reward    │  │  Student + Reward    │
│  GPU 0, GPU 1        │   │  GPU 0, GPU 1        │  │  GPU 0, GPU 1        │
└──────────────────────┘   └──────────────────────┘  └──────────────────────┘

Each Node:
  - 2 GPUs
  - 2 processes (1 per GPU)
  - 1 reward model sidecar (shared by both processes)
  - Independent trajectory collection
  - Synchronized gradient updates
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Flow                                 │
└─────────────────────────────────────────────────────────────────┘

Environment                    Student                  Reward Model
    │                             │                          │
    │ Get prompt                  │                          │
    ├────────────────────────────▶│                          │
    │                             │                          │
    │                             │ Generate response        │
    │                             │ (forward pass)           │
    │                             │                          │
    │                             │ Query reward             │
    │                             ├─────────────────────────▶│
    │                             │                          │
    │                             │                    Score response
    │                             │                          │
    │                             │ Receive reward           │
    │                             │◀─────────────────────────┤
    │                             │                          │
    │ Record episode              │                          │
    │◀────────────────────────────┤                          │
    │                             │                          │
    │                             │ Compute advantages       │
    │                             │ (using rewards)          │
    │                             │                          │
    │                             │ PPO Update               │
    │                             │ (backward pass)          │
    │                             │                          │
    │                             │ Save checkpoint          │
    │                             │ (every N steps)          │
    │                             │                          │
    ▼                             ▼                          ▼

Checkpoints saved to /checkpoints/
- checkpoint-epoch-1/
- checkpoint-epoch-2/
- final/
```

## Resource Allocation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pod Resource Layout                           │
└─────────────────────────────────────────────────────────────────┘

GPU Memory (40GB A100):
┌────────────────────────────────────────────────┐
│  Student Model (GPT-2):         ~6 GB          │
│  Reference Model (frozen):      ~6 GB          │
│  Gradients & Optimizer:         ~4 GB          │
│  Activations & Buffers:         ~4 GB          │
├────────────────────────────────────────────────┤
│  Reward Model:                  ~4 GB          │
│  Buffers:                       ~2 GB          │
├────────────────────────────────────────────────┤
│  Free / Fragmentation:          ~14 GB         │
└────────────────────────────────────────────────┘

RAM (32GB):
┌────────────────────────────────────────────────┐
│  Student Container:             ~12 GB         │
│    - Python runtime: 2GB                       │
│    - Libraries: 4GB                            │
│    - Buffers: 6GB                              │
├────────────────────────────────────────────────┤
│  Reward Model Container:        ~8 GB          │
│    - Python runtime: 2GB                       │
│    - Libraries: 3GB                            │
│    - Buffers: 3GB                              │
├────────────────────────────────────────────────┤
│  System:                        ~12 GB         │
└────────────────────────────────────────────────┘

CPU (16 cores):
┌────────────────────────────────────────────────┐
│  Student Container:             8 cores        │
│  Reward Model Container:        4 cores        │
│  System:                        4 cores        │
└────────────────────────────────────────────────┘
```

## Failure Modes & Recovery

```
┌─────────────────────────────────────────────────────────────────┐
│                    Failure Scenarios                             │
└─────────────────────────────────────────────────────────────────┘

1. Reward Model Crash
   │
   ├─► Kubernetes restarts container (livenessProbe)
   │
   ├─► Student waits for readiness (readinessProbe)
   │
   └─► Training resumes when reward model is ready

2. Student Container OOM
   │
   ├─► Pod gets OOMKilled
   │
   ├─► Job restarts pod (backoffLimit: 2)
   │
   ├─► Student loads latest checkpoint
   │
   └─► Training resumes from checkpoint

3. Node Failure
   │
   ├─► Pod marked as Failed
   │
   ├─► Job creates new pod on different node
   │
   ├─► New pod starts, loads checkpoint from PVC
   │
   └─► Training resumes

4. Network Partition (Multi-Node)
   │
   ├─► NCCL timeout
   │
   ├─► Training fails
   │
   ├─► Job restarts all pods
   │
   └─► Resume from synchronized checkpoint
```

## Security Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│                    Security Architecture                         │
└─────────────────────────────────────────────────────────────────┘

Network:
- Reward model only accessible within pod (localhost)
- No external network access required
- No ingress needed

Containers:
- Non-root user (optional, add to Dockerfile)
- Read-only root filesystem (optional)
- Drop capabilities (optional)

Secrets:
- Image pull secrets for private registries
- Optional: HuggingFace token for model downloads
- Optional: Weights & Biases API key

Resource Limits:
- Memory limits prevent OOM affecting node
- CPU limits prevent resource starvation
- GPU allocation exclusive per pod
```

## Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────────┐
│                    Observability Stack                           │
└─────────────────────────────────────────────────────────────────┘

Metrics Flow:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Student     │────▶│  Prometheus  │────▶│   Grafana    │
│  Container   │     │   (metrics)  │     │  (dashboards)│
└──────────────┘     └──────────────┘     └──────────────┘

Logs Flow:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Both        │────▶│  FluentBit   │────▶│ Elasticsearch│
│  Containers  │     │  (collector) │     │   (storage)  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │    Kibana    │
                                          │    (UI)      │
                                          └──────────────┘

Key Metrics:
- Reward (average, min, max, recent_100)
- Loss (policy, value, entropy)
- KL Divergence
- Throughput (episodes/hour)
- GPU utilization
- Memory usage
- Request latency (reward model)
```

## Deployment Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                                │
└─────────────────────────────────────────────────────────────────┘

Code Changes
     │
     ├─► Build Docker Images
     │   ├─► student-agent:latest
     │   └─► reward-model:latest
     │
     ├─► Run Tests
     │   ├─► Unit tests
     │   ├─► Integration tests
     │   └─► Container smoke tests
     │
     ├─► Push to Registry
     │   └─► docker.io/org/image:tag
     │
     ├─► Deploy to Staging
     │   ├─► Apply TrainingRuntime
     │   ├─► Apply TrainJob
     │   └─► Monitor for 1 epoch
     │
     ├─► Validation
     │   ├─► Check pod status
     │   ├─► Verify metrics
     │   └─► Review logs
     │
     └─► Deploy to Production
         └─► Apply to prod namespace
```

This architecture supports:
- Scalability (1 to 100+ GPUs)
- Reliability (automatic restarts, checkpoints)
- Observability (metrics, logs, traces)
- Security (network isolation, resource limits)
- Maintainability (clean separation of concerns)
