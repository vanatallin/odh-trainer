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

package metrics_test

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/metrics"
)

// Example demonstrates how to record basic TrainJob metrics
func Example_basicMetrics() {
	// Create a metrics recorder
	recorder := metrics.NewMetricsRecorder()

	// Create a sample TrainJob
	trainJob := &trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "example-job",
			Namespace: "default",
		},
		Spec: trainer.TrainJobSpec{
			RuntimeRef: trainer.RuntimeRef{
				Name: "torch-distributed",
			},
		},
	}

	// Record TrainJob creation
	recorder.RecordTrainJobCreation(trainJob)

	// Record phase transition from Created to Running
	recorder.RecordTrainJobPhaseTransition(trainJob, "Created", "Running")

	// Record reconciliation duration
	duration := 150 * time.Millisecond
	recorder.RecordReconciliationDuration(trainJob, duration)

	// Record completion
	trainJob.Status.Conditions = []metav1.Condition{
		{
			Type:               trainer.TrainJobComplete,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
		},
	}
	recorder.RecordTrainJobCompletion(trainJob)
}

// Example demonstrates how to register custom metrics for a plugin
func Example_customMetrics() {
	// Define custom metrics for your plugin
	var (
		customCounter = prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: metrics.TrainerNamespace,
				Subsystem: "myplugin",
				Name:      "operations_total",
				Help:      "Total number of custom operations performed",
			},
			[]string{"operation_type", "status"},
		)

		customGauge = prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: metrics.TrainerNamespace,
				Subsystem: "myplugin",
				Name:      "active_resources",
				Help:      "Current number of active custom resources",
			},
			[]string{"resource_type"},
		)

		customHistogram = prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: metrics.TrainerNamespace,
				Subsystem: "myplugin",
				Name:      "operation_duration_seconds",
				Help:      "Duration of custom operations in seconds",
				Buckets:   prometheus.DefBuckets,
			},
			[]string{"operation_type"},
		)
	)

	// Register all custom metrics
	err := metrics.RegisterCustomMetrics(customCounter, customGauge, customHistogram)
	if err != nil {
		// Handle registration error
		panic(err)
	}

	// Use the custom metrics
	customCounter.With(prometheus.Labels{
		"operation_type": "data_preprocessing",
		"status":         "success",
	}).Inc()

	customGauge.With(prometheus.Labels{
		"resource_type": "dataset",
	}).Set(5)

	customHistogram.With(prometheus.Labels{
		"operation_type": "model_initialization",
	}).Observe(2.5)
}

// Example demonstrates recording reconciliation errors
func Example_reconciliationErrors() {
	recorder := metrics.NewMetricsRecorder()

	trainJob := &trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "example-job",
			Namespace: "default",
		},
		Spec: trainer.TrainJobSpec{
			RuntimeRef: trainer.RuntimeRef{
				Name: "torch-distributed",
			},
		},
	}

	// Record different types of errors
	recorder.RecordReconciliationError(trainJob, "validation_failed")
	recorder.RecordReconciliationError(trainJob, "resource_creation_failed")
	recorder.RecordReconciliationError(trainJob, "runtime_not_found")
}

// Example demonstrates tracking condition transitions
func Example_conditionTransitions() {
	recorder := metrics.NewMetricsRecorder()

	trainJob := &trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "example-job",
			Namespace: "default",
		},
		Spec: trainer.TrainJobSpec{
			RuntimeRef: trainer.RuntimeRef{
				Name: "torch-distributed",
			},
		},
	}

	// Record condition transitions
	recorder.RecordTrainJobConditionTransition(trainJob, trainer.TrainJobSuspended)
	recorder.RecordTrainJobConditionTransition(trainJob, trainer.TrainJobComplete)
	recorder.RecordTrainJobConditionTransition(trainJob, trainer.TrainJobFailed)
}
