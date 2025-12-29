/*
Copyright 2024 The Kubeflow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

const (
	// TrainerNamespace is the Prometheus namespace for all trainer metrics.
	TrainerNamespace = "trainer"

	// Common label names
	TrainJobNameLabel      = "trainjob_name"
	TrainJobNamespaceLabel = "trainjob_namespace"
	TrainJobRuntimeLabel   = "runtime"
	TrainJobConditionLabel = "condition"
	TrainJobPhaseLabel     = "phase"
)

var (
	// TrainJobsTotal tracks the total number of TrainJobs created.
	TrainJobsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: TrainerNamespace,
			Name:      "trainjobs_total",
			Help:      "Total number of TrainJobs created",
		},
		[]string{TrainJobNamespaceLabel, TrainJobRuntimeLabel},
	)

	// TrainJobsActive tracks the current number of active TrainJobs by phase.
	TrainJobsActive = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: TrainerNamespace,
			Name:      "trainjobs_active",
			Help:      "Current number of active TrainJobs by phase (Created, Running, Suspended)",
		},
		[]string{TrainJobNamespaceLabel, TrainJobRuntimeLabel, TrainJobPhaseLabel},
	)

	// TrainJobsCompleted tracks the total number of completed TrainJobs.
	TrainJobsCompleted = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: TrainerNamespace,
			Name:      "trainjobs_completed_total",
			Help:      "Total number of completed TrainJobs",
		},
		[]string{TrainJobNamespaceLabel, TrainJobRuntimeLabel},
	)

	// TrainJobsFailed tracks the total number of failed TrainJobs.
	TrainJobsFailed = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: TrainerNamespace,
			Name:      "trainjobs_failed_total",
			Help:      "Total number of failed TrainJobs",
		},
		[]string{TrainJobNamespaceLabel, TrainJobRuntimeLabel},
	)

	// TrainJobDurationSeconds tracks the duration of TrainJobs from creation to completion/failure.
	TrainJobDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: TrainerNamespace,
			Name:      "trainjob_duration_seconds",
			Help:      "Duration of TrainJobs from creation to completion or failure in seconds",
			Buckets:   []float64{60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400}, // 1min to 24hrs
		},
		[]string{TrainJobNamespaceLabel, TrainJobRuntimeLabel, TrainJobConditionLabel},
	)

	// TrainJobConditionTransitions tracks transitions between TrainJob conditions.
	TrainJobConditionTransitions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: TrainerNamespace,
			Name:      "trainjob_condition_transitions_total",
			Help:      "Total number of TrainJob condition transitions",
		},
		[]string{TrainJobNamespaceLabel, TrainJobRuntimeLabel, TrainJobConditionLabel},
	)

	// TrainJobReconciliationDuration tracks the duration of TrainJob reconciliation operations.
	TrainJobReconciliationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: TrainerNamespace,
			Name:      "trainjob_reconciliation_duration_seconds",
			Help:      "Duration of TrainJob reconciliation operations in seconds",
			Buckets:   prometheus.DefBuckets, // Default buckets: 0.005s to 10s
		},
		[]string{TrainJobNamespaceLabel, TrainJobRuntimeLabel},
	)

	// TrainJobReconciliationErrors tracks errors during TrainJob reconciliation.
	TrainJobReconciliationErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: TrainerNamespace,
			Name:      "trainjob_reconciliation_errors_total",
			Help:      "Total number of errors during TrainJob reconciliation",
		},
		[]string{TrainJobNamespaceLabel, TrainJobRuntimeLabel, "error_type"},
	)
)

// init registers all metrics with the controller-runtime metrics registry.
// This happens automatically when the package is imported.
func init() {
	metrics.Registry.MustRegister(
		TrainJobsTotal,
		TrainJobsActive,
		TrainJobsCompleted,
		TrainJobsFailed,
		TrainJobDurationSeconds,
		TrainJobConditionTransitions,
		TrainJobReconciliationDuration,
		TrainJobReconciliationErrors,
	)
}

// RegisterCustomMetrics allows users to register additional custom metrics.
// This is useful for plugins or extensions that need to expose their own metrics.
func RegisterCustomMetrics(collectors ...prometheus.Collector) error {
	for _, collector := range collectors {
		if err := metrics.Registry.Register(collector); err != nil {
			return err
		}
	}
	return nil
}
