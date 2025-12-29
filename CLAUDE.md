# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kubeflow Trainer is a Kubernetes-native project for distributed training of machine learning models (LLMs, PyTorch, JAX, TensorFlow). It provides a unified Training API that abstracts different distributed training frameworks through a plugin-based runtime system.

**Key Characteristics:**
- Kubernetes operator managing TrainJob, TrainingRuntime, and ClusterTrainingRuntime CRDs
- Plugin-based runtime framework supporting multiple distributed training frameworks (PyTorch, MPI, JAX, etc.)
- Multi-language codebase: Go (operator), Python (initializers), Rust (data cache)
- Integration with JobSet, Volcano, and scheduler-plugins for job orchestration
- Active migration from Training Operator V1 (release-1.9 branch) to V2 (alpha)

## Development Commands

### Building and Code Generation

```bash
# Generate manifests (CRDs, RBAC, webhooks) and Go APIs
make generate

# Download Go modules
make go-mod-download

# Format code
make fmt

# Vet code
make vet

# Lint (golangci-lint)
make golangci-lint
```

### Testing

```bash
# Run Go unit tests (excludes test/, cmd/, hack/, generated code)
make test

# Run Go integration tests (requires ginkgo, envtest, external CRDs)
make test-integration

# Run Python unit tests
make test-python

# Run Python integration tests
make test-python-integration

# Run Rust unit tests (data cache)
make test-rust

# Setup Kind cluster for e2e tests
make test-e2e-setup-cluster

# Run e2e tests
make test-e2e

# Run Jupyter notebook tests with Papermill
make test-e2e-notebook
```

### RHOAI Deployment

This repository includes RHOAI-specific extensions in [pkg/rhai/](pkg/rhai/) and deployment manifests in [manifests/rhoai/](manifests/rhoai/):

```bash
# Deploy to OpenShift/K8s with RHOAI manifests (uses oc or kubectl)
make deploy-rhoai NAMESPACE=opendatahub

# Undeploy
make undeploy-rhoai NAMESPACE=opendatahub
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

### Running a Single Test

```bash
# Go unit test for specific package
go test ./pkg/controller -v -run TestTrainJobController

# Integration test with ginkgo
$(LOCALBIN)/ginkgo -v --focus="TrainingRuntime" ./test/integration/controller/

# Python test
PYTHONPATH=$(pwd) pytest ./pkg/initializers/dataset/cache_test.py -v
```

## Architecture

### Core Components

**Main Entry Point:** [cmd/trainer-controller-manager/main.go](cmd/trainer-controller-manager/main.go:1)
- Initializes controller-runtime manager with configuration from [pkg/config/](pkg/config/)
- Registers schemes: Trainer CRDs, JobSet, Volcano, scheduler-plugins
- Sets up controllers, webhooks, and certificate management

**API Definitions:** [pkg/apis/trainer/v1alpha1/](pkg/apis/trainer/v1alpha1/)
- `TrainJob`: User-facing API for training jobs
- `TrainingRuntime`: Namespace-scoped runtime templates
- `ClusterTrainingRuntime`: Cluster-scoped runtime templates
- Generated clients in [pkg/client/](pkg/client/) (excluded from manual edits)

**Controllers:** [pkg/controller/](pkg/controller/)
- `trainjob_controller.go`: Reconciles TrainJob resources by delegating to runtime plugins
- `trainingruntime_controller.go`: Validates and manages TrainingRuntime lifecycle
- `clustertrainingruntime_controller.go`: Validates and manages ClusterTrainingRuntime lifecycle

**Webhooks:** [pkg/webhooks/](pkg/webhooks/)
- Validation webhooks for all three CRD types
- Certificate rotation handled by cert-controller in [pkg/util/cert/](pkg/util/cert/)

### Runtime Plugin System

**Core Concept:** TrainJob execution is delegated to runtime plugins that implement the [runtime.Runtime](pkg/runtime/interface.go:34) interface.

**Runtime Interface:**
```go
type Runtime interface {
    NewObjects(ctx, trainJob) ([]ApplyConfiguration, error)
    RuntimeInfo(trainJob, templateSpec, mlPolicy, podGroupPolicy) (*Info, error)
    TrainJobStatus(ctx, trainJob) (*TrainJobStatus, error)
    EventHandlerRegistrars() []ReconcilerBuilder
    ValidateObjects(ctx, old, new) (Warnings, ErrorList)
}
```

**Plugin Framework:** [pkg/runtime/framework/](pkg/runtime/framework/)
- `core/framework.go`: Plugin orchestration layer that composes multiple plugins
- `interface.go`: Plugin interface definitions (EnforceMLPolicyPlugin, CustomValidationPlugin, etc.)
- `plugins/`: Concrete plugin implementations
  - `jobset/`: Primary plugin using JobSet for job orchestration
  - `torch/`: PyTorch-specific environment variables and distributed config
  - `mpi/`: MPI runtime support (HPC/collective operations)
  - `plainml/`: Plain ML jobs without distributed training
  - `coscheduling/`: Gang scheduling via scheduler-plugins
  - `volcano/`: Gang scheduling via Volcano

**Plugin Registry:** [pkg/runtime/framework/plugins/registry.go](pkg/runtime/framework/plugins/registry.go:1)
- Registers all available plugins at startup
- Framework instantiates plugins and routes TrainJob reconciliation

**Runtime Resolution:** [pkg/runtime/core/](pkg/runtime/core/)
- `trainingruntime.go`: Resolves runtime templates and applies user overrides
- `clustertrainingruntime.go`: Cluster-level runtime resolution
- Merges TrainingRuntime template with TrainJob spec to produce final JobSet/objects

### Key Abstractions

**PodSet:** [pkg/runtime/runtime.go:62](pkg/runtime/runtime.go:62)
- Abstract representation of multiple PodSpecs as a unit
- Extracted from runtime template objects (JobSet, MPIJob, etc.)
- Contains count, containers, volumes, resource requests

**Info:** [pkg/runtime/runtime.go:36](pkg/runtime/runtime.go:36)
- Runtime metadata (labels, annotations, scheduler config)
- TemplateSpec containing ObjApply and PodSets
- Passed to plugins for policy enforcement

**MLPolicy and PodGroupPolicy:**
- MLPolicy: NumNodes, numProcPerNode, TorchElasticPolicy
- PodGroupPolicy: Gang scheduling configuration
- Enforced by plugins through `EnforceMLPolicyPlugin` and `EnforcePodGroupPolicyPlugin` interfaces

### Initializers (Python)

**Location:** [pkg/initializers/](pkg/initializers/)
- `dataset/`: Dataset loading (HuggingFace, S3, local storage)
- `model/`: Model initialization (HuggingFace, torch, etc.)
- `utils/`: Common utilities for initializers

**Usage:** Run as init containers in training pods to prepare data/models before training starts.

**Testing:** Python tests use pytest ([pkg/initializers/conftest.py](pkg/initializers/conftest.py:1))

### Data Cache (Rust)

**Location:** [pkg/data_cache/](pkg/data_cache/)
- Distributed data caching system for training datasets
- Head/worker architecture with gRPC communication
- Supports S3 backends and IAM role assumption

**Build:**
```bash
cargo build --manifest-path ./pkg/data_cache/Cargo.toml
```

### RHOAI Extensions

**Location:** [pkg/rhai/](pkg/rhai/)
- Midstream-only features not in upstream Kubeflow
- `progression/`: Real-time training metrics polling and status updates
- `constants/`: RHOAI-specific annotations and constants
- Enabled per-TrainJob via `trainer.opendatahub.io/progression-tracking: "enabled"` annotation
- Integrated in `TrainJobReconciler` with minimal coupling

## Important Conventions

### Code Generation

**DO NOT manually edit generated files:**
- [pkg/client/](pkg/client/)
- [api/python_api/kubeflow_trainer_api/models/](api/python_api/kubeflow_trainer_api/models/)
- Files with `zz_generated` prefix

**Regenerate after API changes:**
```bash
make generate
```

### Testing Conventions

**Go Tests:**
- Use `cmp.Diff` instead of `reflect.Equal` for comparisons
- Define test cases as maps (not slices) to avoid order dependencies
- Map keys should equal test case names

**Integration Tests:**
- Use Ginkgo/Gomega testing framework
- External CRDs required (JobSet, scheduler-plugins, Volcano)
- Download with: `make jobset-operator-crd scheduler-plugins-crd volcano-crd`

### Pull Request Titles

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `fix(operator): Check empty value for registry`
- `feat(docs): Create guide for LLM Fine-Tuning`
- `chore(ci): Remove unused scripts`

Valid types/scopes defined in [.github/workflows/check-pr-title.yaml](.github/workflows/check-pr-title.yaml:1)

### Pre-commit Hooks

Install and run before commits:
```bash
pip install pre-commit
pre-commit install
```

Hooks enforce:
- Go fmt/vet (implicitly via golangci-lint in CI)
- Python: isort, black, flake8
- Rust: cargo fmt, cargo check (data_cache)
- YAML/JSON validation

## Runtime Template Flow

1. User creates TrainJob referencing a TrainingRuntime (or ClusterTrainingRuntime)
2. Controller resolves runtime template from TrainingRuntime CRD
3. Runtime core merges template with TrainJob spec overrides
4. Runtime plugin framework applies policies:
   - EnforceMLPolicy (num nodes, procs per node)
   - EnforcePodGroupPolicy (gang scheduling)
   - CustomValidation (runtime-specific validation)
   - ComponentBuilder (build final objects like JobSet, ConfigMaps)
5. Plugin returns ApplyConfigurations (e.g., JobSet, PodGroup)
6. Controller applies objects to cluster
7. Plugin watches for events (via EventHandlerRegistrars)
8. Plugin computes TrainJobStatus from underlying objects

## Key Files to Reference

- **Controller setup:** [pkg/controller/setup.go](pkg/controller/setup.go:1)
- **Runtime resolution:** [pkg/runtime/core/trainingruntime.go](pkg/runtime/core/trainingruntime.go:1)
- **JobSet plugin:** [pkg/runtime/framework/plugins/jobset/](pkg/runtime/framework/plugins/jobset/)
- **Webhook registration:** [pkg/webhooks/setup.go](pkg/webhooks/setup.go:1)
- **Configuration API:** [pkg/apis/config/v1alpha1/](pkg/apis/config/v1alpha1/)
- **Examples:** [examples/](examples/) (PyTorch, DeepSpeed, MLX, Torchtune)

## External Dependencies

- **JobSet:** Primary job orchestration primitive (replaces individual framework operators)
- **Volcano:** Optional gang scheduler for batch scheduling
- **scheduler-plugins:** Optional coscheduling plugin
- **cert-controller:** Webhook certificate management

Install external CRDs for local testing:
```bash
make jobset-operator-crd scheduler-plugins-crd volcano-crd
```
