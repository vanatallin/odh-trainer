# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Kubeflow Trainer V2 (odh-trainer fork for RHOAI), a Kubernetes-native platform for distributed training of machine learning models (PyTorch, JAX, TensorFlow, etc.) with focus on LLM fine-tuning. The project uses Kubernetes Custom Resource Definitions (CRDs) and a controller-based architecture.

Core CRDs:
- **TrainJob**: Represents a training job instance
- **TrainingRuntime/ClusterTrainingRuntime**: Defines how training jobs are executed (namespace-scoped and cluster-scoped)

## Development Commands

### Building and Code Generation

```bash
# Generate manifests (CRDs, RBAC, webhooks) and Go APIs
make generate

# Format Go code
make fmt

# Run Go vet
make vet

# Run golangci-lint
make golangci-lint
```

### Testing

```bash
# Run Go unit tests
make test

# Run Python unit tests
make test-python

# Run Go integration tests (sets up envtest with external CRDs)
make test-integration

# Run Python integration tests
make test-python-integration

# Run Rust unit tests (for data cache component)
make test-rust

# Setup Kind cluster for e2e tests
make test-e2e-setup-cluster

# Run e2e tests (requires cluster setup first)
make test-e2e

# Run Jupyter notebook test with Papermill
make test-e2e-notebook
```

### Helm

```bash
# Run Helm chart unit tests
make helm-unittest

# Lint Helm charts
make helm-lint

# Generate Helm documentation
make helm-docs
```

### RHOAI Deployment

```bash
# Deploy operator to OpenShift/K8s using RHOAI manifests
make deploy-rhoai NAMESPACE=opendatahub

# Undeploy operator
make undeploy-rhoai NAMESPACE=opendatahub

# Use KUBECTL variable to specify oc or kubectl
make deploy-rhoai KUBECTL=oc NAMESPACE=my-namespace
```

## Architecture

### Controller Architecture

The project follows Kubernetes operator patterns using controller-runtime:

1. **Main Controller** (`cmd/trainer-controller-manager/main.go`): Entry point that sets up:
   - Manager with client, cache, and webhook server
   - TrainJob controller
   - TrainingRuntime/ClusterTrainingRuntime controllers
   - Validation webhooks

2. **Controllers** (`pkg/controller/`):
   - `trainjob_controller.go`: Reconciles TrainJob resources
   - `trainingruntime_controller.go`: Manages TrainingRuntime resources
   - `clustertrainingruntime_controller.go`: Manages ClusterTrainingRuntime resources

3. **Webhooks** (`pkg/webhooks/`): Admission webhooks for validation and defaulting

### Runtime Plugin System

The project uses a plugin-based runtime system (`pkg/runtime/`):

- **Runtime Interface** (`pkg/runtime/interface.go`): Defines contract for creating and managing training jobs
- **Runtime Core** (`pkg/runtime/core/`): Registry and base implementations for TrainingRuntime/ClusterTrainingRuntime
- **Framework Plugins** (`pkg/runtime/framework/`):
  - Plugin interface defines extension points
  - Plugin types: CustomValidation, WatchExtension, EnforcePodGroupPolicy, EnforceMLPolicy, PodNetwork, ComponentBuilder, TrainJobStatus
  - Concrete plugins in `pkg/runtime/framework/plugins/`:
    - `jobset/`: Kubernetes JobSet integration (primary execution runtime)
    - `mpi/`: MPI-based training support
    - `torch/`: PyTorch-specific features
    - `coscheduling/`: Gang scheduling support (Volcano, scheduler-plugins)

### Key Components

- **APIs** (`pkg/apis/trainer/v1alpha1/`):
  - `trainjob_types.go`: TrainJob CRD definition
  - `trainingruntime_types.go`: TrainingRuntime/ClusterTrainingRuntime CRD definitions

- **Initializers** (`pkg/initializers/`): Dataset and model initialization logic (Python)

- **Data Cache** (`pkg/data_cache/`): Rust-based data caching component

- **RHOAI Extensions** (`pkg/rhai/`): Red Hat OpenShift AI specific features
  - `progression/`: Training progression tracking

### Directory Structure

```
cmd/                          # Main binaries
  trainer-controller-manager/ # Controller manager entry point
  initializers/               # Dataset/model initializer scripts
  runtimes/                   # Runtime-specific tools
  trainers/                   # Training executors
manifests/                    # Kubernetes manifests
  base/                       # Base Kustomize overlays (CRDs, RBAC, webhooks)
  rhoai/                      # RHOAI-specific manifests and runtimes
  overlays/                   # Environment-specific overlays
pkg/                          # Go packages
  apis/                       # API definitions (CRDs)
  controller/                 # Controller implementations
  runtime/                    # Runtime plugin system
  webhooks/                   # Admission webhooks
  initializers/               # Python initializer utilities
  data_cache/                 # Rust data caching (Cargo project)
  rhai/                       # RHOAI-specific features
test/                         # Tests
  integration/                # Integration tests (Go + Python)
  e2e/                        # End-to-end tests
examples/                     # Example training jobs (PyTorch, DeepSpeed, etc.)
charts/kubeflow-trainer/      # Helm chart
```

## Code Style and Conventions

### Pre-commit Hooks

Install pre-commit hooks before committing:
```bash
pip install pre-commit
pre-commit install
```

Hooks run on staged files automatically. To run on all files:
```bash
pre-commit run --all
```

### PR Title Convention

Follow Conventional Commits specification with type and scope:
- `feat(operator): Add new runtime plugin`
- `fix(webhook): Validate training runtime reference`
- `chore(ci): Update test workflows`

### Go Development

- Follow [Effective Go](https://go.dev/doc/effective_go) guidelines
- Use `cmp.Diff()` instead of `reflect.Equal()` in tests
- Define test cases as maps (not slices) to avoid order dependencies
- Run `make generate` before submitting PRs to regenerate code

### Testing Patterns

- Unit tests: Test individual functions/packages (`*_test.go` files)
- Integration tests: Use envtest (controller-runtime test environment) with external CRDs
- E2E tests: Use Ginkgo framework against real Kind cluster

## External Dependencies

The project integrates with external Kubernetes CRDs:
- **JobSet** (sigs.k8s.io/jobset): Primary job execution runtime
- **Scheduler Plugins** (sigs.k8s.io/scheduler-plugins): Coscheduling support
- **Volcano** (volcano.sh/apis): Alternative gang scheduling

External CRDs are copied to `manifests/external-crds/` via:
```bash
make jobset-operator-crd
make scheduler-plugins-crd
make volcano-crd
```

## RHOAI Specific

This fork includes Red Hat OpenShift AI (RHOAI) specific customizations in `manifests/rhoai/`:
- Custom training runtimes in `manifests/rhoai/runtimes/`
- RHOAI-specific configuration patches
- Integration with OpenDataHub components

When working with RHOAI deployments, use `make deploy-rhoai` which handles CRD installation ordering and proper server-side apply.
