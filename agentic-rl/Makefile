# Makefile for Agentic RL Training

# Configuration
REGISTRY ?= quay.io/umb-man-rh/pocs
STUDENT_IMAGE := $(REGISTRY)/student-agent
REWARD_IMAGE := $(REGISTRY)/reward-model
TAG ?= 0.1

# Kubernetes
NAMESPACE ?= default
RUNTIME_NAME ?= agentic-rl-pytorch
JOB_NAME ?= agentic-rl-training

.PHONY: help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: build
build: build-student build-reward ## Build all Docker images

.PHONY: build-student
build-student: ## Build student agent Docker image
	@echo "Building student agent image..."
	cd student && podman build -t $(STUDENT_IMAGE):$(TAG) .

.PHONY: build-reward
build-reward: ## Build reward model Docker image
	@echo "Building reward model image..."
	cd reward-model && podman build -t $(REWARD_IMAGE):$(TAG) .

.PHONY: push
push: push-student push-reward ## Push all Docker images

.PHONY: push-student
push-student: ## Push student agent Docker image
	@echo "Pushing student agent image..."
	podman push $(STUDENT_IMAGE):$(TAG)

.PHONY: push-reward
push-reward: ## Push reward model Docker image
	@echo "Pushing reward model image..."
	podman push $(REWARD_IMAGE):$(TAG)

.PHONY: deploy-runtime
deploy-runtime: ## Deploy TrainingRuntime
	@echo "Deploying TrainingRuntime..."
	kubectl apply -f manifests/training-runtime.yaml -n $(NAMESPACE)

.PHONY: deploy-cluster-runtime
deploy-cluster-runtime: ## Deploy ClusterTrainingRuntime
	@echo "Deploying ClusterTrainingRuntime..."
	kubectl apply -f manifests/cluster-training-runtime.yaml

.PHONY: deploy-job
deploy-job: ## Deploy TrainJob
	@echo "Deploying TrainJob..."
	@# Update image references in train-job.yaml
	@sed -e 's|<your-registry>/student-agent:latest|$(STUDENT_IMAGE):$(TAG)|g' \
	     -e 's|<your-registry>/reward-model:latest|$(REWARD_IMAGE):$(TAG)|g' \
	     manifests/train-job.yaml | kubectl apply -f - -n $(NAMESPACE)

.PHONY: delete-job
delete-job: ## Delete TrainJob
	@echo "Deleting TrainJob..."
	kubectl delete trainjob $(JOB_NAME) -n $(NAMESPACE) --ignore-not-found

.PHONY: delete-runtime
delete-runtime: ## Delete TrainingRuntime
	@echo "Deleting TrainingRuntime..."
	kubectl delete trainingruntime $(RUNTIME_NAME) -n $(NAMESPACE) --ignore-not-found

.PHONY: logs
logs: ## View training logs
	@echo "Fetching logs from student container..."
	@POD=$$(kubectl get pods -n $(NAMESPACE) -l trainjob-name=$(JOB_NAME) -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -n "$$POD" ]; then \
		kubectl logs -f $$POD -c node -n $(NAMESPACE); \
	else \
		echo "No pod found for job $(JOB_NAME)"; \
	fi

.PHONY: logs-reward
logs-reward: ## View reward model logs
	@echo "Fetching logs from reward model container..."
	@POD=$$(kubectl get pods -n $(NAMESPACE) -l trainjob-name=$(JOB_NAME) -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -n "$$POD" ]; then \
		kubectl logs -f $$POD -c reward-model -n $(NAMESPACE); \
	else \
		echo "No pod found for job $(JOB_NAME)"; \
	fi

.PHONY: status
status: ## Check TrainJob status
	@echo "Checking TrainJob status..."
	@kubectl get trainjob $(JOB_NAME) -n $(NAMESPACE) -o wide || echo "Job not found"
	@echo ""
	@echo "Pods:"
	@kubectl get pods -n $(NAMESPACE) -l trainjob-name=$(JOB_NAME)

.PHONY: describe
describe: ## Describe TrainJob
	@kubectl describe trainjob $(JOB_NAME) -n $(NAMESPACE)

.PHONY: exec-student
exec-student: ## Execute shell in student container
	@POD=$$(kubectl get pods -n $(NAMESPACE) -l trainjob-name=$(JOB_NAME) -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -n "$$POD" ]; then \
		kubectl exec -it $$POD -c node -n $(NAMESPACE) -- /bin/bash; \
	else \
		echo "No pod found for job $(JOB_NAME)"; \
	fi

.PHONY: exec-reward
exec-reward: ## Execute shell in reward model container
	@POD=$$(kubectl get pods -n $(NAMESPACE) -l trainjob-name=$(JOB_NAME) -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -n "$$POD" ]; then \
		kubectl exec -it $$POD -c reward-model -n $(NAMESPACE) -- /bin/bash; \
	else \
		echo "No pod found for job $(JOB_NAME)"; \
	fi

.PHONY: test-reward
test-reward: ## Test reward model health
	@echo "Testing reward model health..."
	@POD=$$(kubectl get pods -n $(NAMESPACE) -l trainjob-name=$(JOB_NAME) -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -n "$$POD" ]; then \
		kubectl exec $$POD -c node -n $(NAMESPACE) -- curl -s http://localhost:8080/health | jq .; \
	else \
		echo "No pod found for job $(JOB_NAME)"; \
	fi

.PHONY: clean
clean: delete-job ## Clean up all resources

.PHONY: all
all: build push deploy-runtime deploy-job ## Build, push, and deploy everything

.PHONY: quick-start
quick-start: ## Quick start guide
	@echo "Quick Start Guide:"
	@echo "1. Set your registry: export REGISTRY=docker.io/your-username"
	@echo "2. Build images:     make build"
	@echo "3. Push images:      make push"
	@echo "4. Deploy runtime:   make deploy-runtime"
	@echo "5. Deploy job:       make deploy-job"
	@echo "6. View logs:        make logs"
	@echo ""
	@echo "Or run everything:   make all"
