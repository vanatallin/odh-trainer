# Troubleshooting Guide

## Common Issues When Using OpenShift AI Workbench

### Error: Certificate Verification Failed

**Full Error:**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

**Cause:**
OpenShift uses self-signed certificates or internal CA certificates that aren't in the default trust store.

**Solution:**

Option 1: Disable SSL verification (development only):
```python
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# For kubectl/oc
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

Option 2: Add CA certificate to trust store:
```bash
# In workbench terminal
oc get secret/router-ca -n openshift-ingress-operator -o jsonpath='{.data.tls\.crt}' | base64 -d > /tmp/ca.crt
export REQUESTS_CA_BUNDLE=/tmp/ca.crt
```

### Error: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'kubeflow.trainer'
```

**Solution:**
Install the Kubeflow Training SDK:
```bash
pip install kubeflow-training
```

### Error: TrainJob Webhook Validation Failed

**Error:**
```
Error from server: failed calling webhook "validator.trainjob.trainer.kubeflow.org":
Post "https://kubeflow-trainer-controller-manager.opendatahub.svc:443/validate-trainer-kubeflow-org-v1alpha1-trainjob":
no endpoints available for service "kubeflow-trainer-controller-manager"
```

**Cause:**
1. Trainer controller not running
2. JobSet CRD missing (required dependency)

**Solution:**
```bash
# Check if JobSet is installed
oc get crd jobsets.jobset.x-k8s.io

# If not found, install JobSet
oc apply -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.10.1/manifests.yaml

# Check trainer controller
oc get pods -n opendatahub -l app.kubernetes.io/name=trainer

# If controller is not running, check logs
oc logs -n opendatahub -l app.kubernetes.io/name=trainer
```

### Error: Namespace Not Found

**Error:**
```
Error from server (NotFound): namespaces "default" not found
```

**Solution:**
Use your actual Data Science Project namespace:
```python
sdk = AgenticRLTrainingSDK(
    namespace='your-project-name',  # Not 'default'
    student_image='...',
    reward_model_image='...'
)
```

Find your namespace:
```bash
oc project  # Shows current project
oc get projects  # Lists all projects
```

### Error: ImagePullBackOff

**Error:**
```
Pod status: ImagePullBackOff
Events: Failed to pull image "your-registry/student-agent:latest": rpc error: code = Unknown desc = Error reading manifest latest in docker.io/your-registry/student-agent: errors
```

**Causes:**
1. Image doesn't exist in registry
2. Image name/tag incorrect
3. Registry requires authentication

**Solutions:**

1. Verify image exists:
```bash
podman search quay.io/your-username/student-agent
```

2. Create image pull secret (for private registries):
```bash
oc create secret docker-registry registry-secret \
  --docker-server=quay.io \
  --docker-username=your-username \
  --docker-password=your-token \
  -n your-namespace
```

3. Use OpenShift internal registry:
```bash
# Get internal registry URL
REGISTRY=$(oc get route default-route -n openshift-image-registry -o jsonpath='{.spec.host}')

# Build and push
podman build -t ${REGISTRY}/your-namespace/student-agent:latest student/
podman push ${REGISTRY}/your-namespace/student-agent:latest
```

### Error: Insufficient Resources

**Error:**
```
0/N nodes are available: N Insufficient nvidia.com/gpu
```

**Cause:**
No GPU nodes available or all GPUs in use.

**Solution:**
See [MACHINE_POOLS_GUIDE.md](MACHINE_POOLS_GUIDE.md) for creating GPU machine pools.

Quick fix - use CPU instead:
```python
job_id = sdk.create_train_job(
    name="my-job",
    gpu_per_node=0,  # No GPU
    cpu_per_node=8,
    memory_per_node="16Gi"
)
```

### Error: Permission Denied

**Error:**
```
Error from server (Forbidden): trainjobs.trainer.kubeflow.org is forbidden:
User "system:serviceaccount:your-namespace:default" cannot create resource "trainjobs"
```

**Solution:**
Grant necessary permissions:
```bash
# For namespace-scoped access
oc adm policy add-role-to-user edit system:serviceaccount:your-namespace:default -n your-namespace

# Or use the workbench's service account
# (OpenShift AI workbenches usually have correct permissions already)
```

### Error: TrainJob Stuck in Pending

**Check pod status:**
```bash
# Find the job's pods
oc get pods -l trainjob-name=your-job-name

# Describe the pod to see why it's pending
oc describe pod <pod-name>
```

**Common causes:**
1. Insufficient resources (CPU/memory/GPU)
2. Node selector doesn't match any nodes
3. Tolerations missing for tainted nodes
4. Image pull errors

### Error: Python Function Not Serializable

**Error:**
```
TypeError: cannot pickle 'module' object
```

**Cause:**
The training function passed to `CustomTrainer` references objects that can't be serialized.

**Solution:**
Keep the function self-contained:
```python
def train_model():
    # Import INSIDE the function
    import torch
    from transformers import AutoModel

    # Don't reference outer scope variables
    model_name = os.getenv('MODEL_NAME', 'default-model')

    # Training code here
    pass

# Pass the function (not called!)
trainer=CustomTrainer(func=train_model)  # ✅ Correct
trainer=CustomTrainer(func=train_model())  # ❌ Wrong
```

## Getting Help

If you're still stuck:

1. Check the logs:
```bash
# Training job logs
oc get trainjob -n your-namespace
oc logs -l trainjob-name=your-job -c node -n your-namespace

# Controller logs
oc logs -n opendatahub -l app.kubernetes.io/name=trainer
```

2. Check events:
```bash
oc get events -n your-namespace --sort-by='.lastTimestamp'
```

3. Verify prerequisites:
```bash
# JobSet installed?
oc get crd jobsets.jobset.x-k8s.io

# Trainer controller running?
oc get pods -n opendatahub -l app.kubernetes.io/name=trainer

# TrainingRuntime exists?
oc get trainingruntimes -n your-namespace
```

4. Open an issue:
   - [GitHub Issues](https://github.com/opendatahub-io/odh-trainer/issues)
   - Include: error message, pod describe output, controller logs
