# Kubeflow Trainer Metrics

This package provides Prometheus metrics for monitoring TrainJob operations in Kubeflow Trainer.

## Overview

The metrics package integrates with the controller-runtime metrics server to expose TrainJob-specific metrics. All metrics are automatically registered and available on the controller manager's metrics endpoint (default: `:8443/metrics`).

## Available Metrics

### TrainJob Creation and Lifecycle

#### `trainer_trainjobs_total`
**Type:** Counter
**Description:** Total number of TrainJobs created
**Labels:**
- `trainjob_namespace`: Namespace of the TrainJob
- `runtime`: Name of the TrainingRuntime referenced

**Example:**
```
trainer_trainjobs_total{trainjob_namespace="default",runtime="torch-distributed"} 42
```

#### `trainer_trainjobs_active`
**Type:** Gauge
**Description:** Current number of active TrainJobs by phase
**Labels:**
- `trainjob_namespace`: Namespace of the TrainJob
- `runtime`: Name of the TrainingRuntime referenced
- `phase`: Current phase (Created, Running, Suspended)

**Example:**
```
trainer_trainjobs_active{trainjob_namespace="default",runtime="torch-distributed",phase="Running"} 5
trainer_trainjobs_active{trainjob_namespace="default",runtime="torch-distributed",phase="Suspended"} 2
```

### Completion and Failure Tracking

#### `trainer_trainjobs_completed_total`
**Type:** Counter
**Description:** Total number of completed TrainJobs
**Labels:**
- `trainjob_namespace`: Namespace of the TrainJob
- `runtime`: Name of the TrainingRuntime referenced

**Example:**
```
trainer_trainjobs_completed_total{trainjob_namespace="default",runtime="torch-distributed"} 38
```

#### `trainer_trainjobs_failed_total`
**Type:** Counter
**Description:** Total number of failed TrainJobs
**Labels:**
- `trainjob_namespace`: Namespace of the TrainJob
- `runtime`: Name of the TrainingRuntime referenced

**Example:**
```
trainer_trainjobs_failed_total{trainjob_namespace="default",runtime="torch-distributed"} 3
```

### Duration Metrics

#### `trainer_trainjob_duration_seconds`
**Type:** Histogram
**Description:** Duration of TrainJobs from creation to completion or failure
**Labels:**
- `trainjob_namespace`: Namespace of the TrainJob
- `runtime`: Name of the TrainingRuntime referenced
- `condition`: Terminal condition (Complete or Failed)

**Buckets:** 60s, 300s (5m), 600s (10m), 1800s (30m), 3600s (1h), 7200s (2h), 14400s (4h), 28800s (8h), 86400s (24h)

**Example:**
```
trainer_trainjob_duration_seconds_bucket{trainjob_namespace="default",runtime="torch-distributed",condition="Complete",le="3600"} 15
trainer_trainjob_duration_seconds_sum{trainjob_namespace="default",runtime="torch-distributed",condition="Complete"} 124532.5
trainer_trainjob_duration_seconds_count{trainjob_namespace="default",runtime="torch-distributed",condition="Complete"} 38
```

### Condition Transitions

#### `trainer_trainjob_condition_transitions_total`
**Type:** Counter
**Description:** Total number of TrainJob condition transitions
**Labels:**
- `trainjob_namespace`: Namespace of the TrainJob
- `runtime`: Name of the TrainingRuntime referenced
- `condition`: Condition type (Suspended, Complete, Failed)

**Example:**
```
trainer_trainjob_condition_transitions_total{trainjob_namespace="default",runtime="torch-distributed",condition="Complete"} 38
trainer_trainjob_condition_transitions_total{trainjob_namespace="default",runtime="torch-distributed",condition="Suspended"} 12
```

### Reconciliation Metrics

#### `trainer_trainjob_reconciliation_duration_seconds`
**Type:** Histogram
**Description:** Duration of TrainJob reconciliation operations
**Labels:**
- `trainjob_namespace`: Namespace of the TrainJob
- `runtime`: Name of the TrainingRuntime referenced

**Buckets:** Default Prometheus buckets (0.005s to 10s)

**Example:**
```
trainer_trainjob_reconciliation_duration_seconds_bucket{trainjob_namespace="default",runtime="torch-distributed",le="0.1"} 1250
trainer_trainjob_reconciliation_duration_seconds_sum{trainjob_namespace="default",runtime="torch-distributed"} 125.3
trainer_trainjob_reconciliation_duration_seconds_count{trainjob_namespace="default",runtime="torch-distributed"} 1500
```

#### `trainer_trainjob_reconciliation_errors_total`
**Type:** Counter
**Description:** Total number of errors during TrainJob reconciliation
**Labels:**
- `trainjob_namespace`: Namespace of the TrainJob
- `runtime`: Name of the TrainingRuntime referenced
- `error_type`: Type of error (runtime_not_found, reconcile_objects_failed)

**Example:**
```
trainer_trainjob_reconciliation_errors_total{trainjob_namespace="default",runtime="torch-distributed",error_type="reconcile_objects_failed"} 2
```

## Usage

### Accessing Metrics

By default, metrics are exposed on the controller manager's metrics endpoint. The endpoint is configured via the `--metrics-bind-address` flag:

```bash
# HTTPS (default)
curl -k https://localhost:8443/metrics

# HTTP (if --metrics-secure=false)
curl http://localhost:8080/metrics
```

### Prometheus Configuration

Add the following to your Prometheus scrape configuration:

```yaml
scrape_configs:
  - job_name: 'kubeflow-trainer'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - kubeflow
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_control_plane]
        action: keep
        regex: controller-manager
      - source_labels: [__meta_kubernetes_pod_container_port_name]
        action: keep
        regex: metrics
```

### Example Queries

**Success rate:**
```promql
sum(rate(trainer_trainjobs_completed_total[5m])) /
(sum(rate(trainer_trainjobs_completed_total[5m])) + sum(rate(trainer_trainjobs_failed_total[5m])))
```

**Average job duration:**
```promql
rate(trainer_trainjob_duration_seconds_sum{condition="Complete"}[5m]) /
rate(trainer_trainjob_duration_seconds_count{condition="Complete"}[5m])
```

**Jobs running per namespace:**
```promql
sum(trainer_trainjobs_active{phase="Running"}) by (trainjob_namespace)
```

**P95 reconciliation latency:**
```promql
histogram_quantile(0.95,
  sum(rate(trainer_trainjob_reconciliation_duration_seconds_bucket[5m])) by (le, runtime)
)
```

**Error rate by type:**
```promql
sum(rate(trainer_trainjob_reconciliation_errors_total[5m])) by (error_type)
```

## Extending Metrics

### Adding Custom Metrics

You can add custom metrics from plugins or extensions using the `RegisterCustomMetrics` function:

```go
package myplugin

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/kubeflow/trainer/v2/pkg/metrics"
)

var (
    myCustomCounter = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Namespace: "trainer",
            Subsystem: "myplugin",
            Name:      "custom_operations_total",
            Help:      "Total number of custom operations",
        },
        []string{"operation_type"},
    )
)

func init() {
    // Register custom metrics during initialization
    if err := metrics.RegisterCustomMetrics(myCustomCounter); err != nil {
        panic(err)
    }
}

func DoCustomOperation(opType string) {
    // Record your custom metric
    myCustomCounter.With(prometheus.Labels{"operation_type": opType}).Inc()
}
```

### Using MetricsRecorder

The `MetricsRecorder` provides helper methods for common metric recording operations:

```go
import (
    "github.com/kubeflow/trainer/v2/pkg/metrics"
    trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
)

recorder := metrics.NewMetricsRecorder()

// Record TrainJob creation
recorder.RecordTrainJobCreation(trainJob)

// Record phase transition
recorder.RecordTrainJobPhaseTransition(trainJob, "Created", "Running")

// Record condition transition
recorder.RecordTrainJobConditionTransition(trainJob, "Suspended")

// Record completion
recorder.RecordTrainJobCompletion(trainJob)

// Record failure
recorder.RecordTrainJobFailure(trainJob)

// Record reconciliation duration
recorder.RecordReconciliationDuration(trainJob, duration)

// Record reconciliation error
recorder.RecordReconciliationError(trainJob, "validation_failed")
```

## Grafana Dashboards

Example Grafana dashboard panels:

### TrainJob Success Rate
```json
{
  "targets": [
    {
      "expr": "sum(rate(trainer_trainjobs_completed_total[5m])) / (sum(rate(trainer_trainjobs_completed_total[5m])) + sum(rate(trainer_trainjobs_failed_total[5m])))"
    }
  ],
  "title": "TrainJob Success Rate"
}
```

### Active TrainJobs by Runtime
```json
{
  "targets": [
    {
      "expr": "sum(trainer_trainjobs_active) by (runtime, phase)"
    }
  ],
  "title": "Active TrainJobs by Runtime"
}
```

### Job Duration Heatmap
```json
{
  "targets": [
    {
      "expr": "sum(increase(trainer_trainjob_duration_seconds_bucket[5m])) by (le)",
      "format": "heatmap"
    }
  ],
  "title": "Job Duration Distribution"
}
```

## Best Practices

1. **Label Cardinality:** Avoid adding labels with high cardinality (e.g., job names, timestamps) to prevent excessive memory usage
2. **Metric Naming:** Follow Prometheus naming conventions: `<namespace>_<subsystem>_<name>_<unit>`
3. **Counter vs Gauge:** Use counters for values that only increase, gauges for values that can go up or down
4. **Histogram Buckets:** Choose buckets appropriate for your use case; default buckets may not be suitable for all metrics
5. **Recording Rules:** Use Prometheus recording rules to pre-compute expensive queries

## Troubleshooting

### Metrics Not Appearing

1. **Check metrics endpoint is accessible:**
   ```bash
   kubectl port-forward -n kubeflow svc/kubeflow-trainer-controller-manager 8443:8443
   curl -k https://localhost:8443/metrics | grep trainer_
   ```

2. **Verify metrics are registered:**
   Check controller manager logs for any metric registration errors during startup

3. **Check Prometheus scraping:**
   Verify Prometheus is successfully scraping the endpoint:
   ```promql
   up{job="kubeflow-trainer"}
   ```

### High Memory Usage

If metrics are consuming excessive memory:
1. Check for high-cardinality labels
2. Reduce histogram bucket counts
3. Use recording rules to aggregate metrics
4. Consider metric retention policies

## Further Reading

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Kubernetes Metrics](https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/)
- [Controller Runtime Metrics](https://book.kubebuilder.io/reference/metrics.html)
