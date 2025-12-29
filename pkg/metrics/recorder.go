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
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
)

// MetricsRecorder provides helper methods for recording TrainJob metrics.
type MetricsRecorder struct{}

// NewMetricsRecorder creates a new MetricsRecorder.
func NewMetricsRecorder() *MetricsRecorder {
	return &MetricsRecorder{}
}

// RecordTrainJobCreation records metrics when a TrainJob is created.
func (r *MetricsRecorder) RecordTrainJobCreation(trainJob *trainer.TrainJob) {
	labels := r.getBaseLabels(trainJob)
	TrainJobsTotal.With(labels).Inc()
}

// RecordTrainJobPhaseTransition records metrics when a TrainJob transitions to a new phase.
// This updates the active gauge metrics.
func (r *MetricsRecorder) RecordTrainJobPhaseTransition(trainJob *trainer.TrainJob, oldPhase, newPhase string) {
	labels := r.getBaseLabels(trainJob)

	// Decrement old phase if it was being tracked
	if oldPhase != "" {
		oldLabels := r.withPhase(labels, oldPhase)
		TrainJobsActive.With(oldLabels).Dec()
	}

	// Increment new phase if it should be tracked
	if newPhase != "" && r.isActivePhase(newPhase) {
		newLabels := r.withPhase(labels, newPhase)
		TrainJobsActive.With(newLabels).Inc()
	}
}

// RecordTrainJobConditionTransition records when a TrainJob condition changes.
func (r *MetricsRecorder) RecordTrainJobConditionTransition(trainJob *trainer.TrainJob, conditionType string) {
	labels := r.getBaseLabels(trainJob)
	labels[TrainJobConditionLabel] = conditionType
	TrainJobConditionTransitions.With(labels).Inc()
}

// RecordTrainJobCompletion records metrics when a TrainJob completes successfully.
func (r *MetricsRecorder) RecordTrainJobCompletion(trainJob *trainer.TrainJob) {
	labels := r.getBaseLabels(trainJob)
	TrainJobsCompleted.With(labels).Inc()

	// Record duration if we have creation and completion timestamps
	if duration := r.calculateDuration(trainJob, trainer.TrainJobComplete); duration > 0 {
		durationLabels := r.withCondition(labels, trainer.TrainJobComplete)
		TrainJobDurationSeconds.With(durationLabels).Observe(duration)
	}

	// Decrement active count if the job was tracked as active
	if phase := r.getCurrentPhase(trainJob); r.isActivePhase(phase) {
		activeLabels := r.withPhase(labels, phase)
		TrainJobsActive.With(activeLabels).Dec()
	}
}

// RecordTrainJobFailure records metrics when a TrainJob fails.
func (r *MetricsRecorder) RecordTrainJobFailure(trainJob *trainer.TrainJob) {
	labels := r.getBaseLabels(trainJob)
	TrainJobsFailed.With(labels).Inc()

	// Record duration if we have creation and failure timestamps
	if duration := r.calculateDuration(trainJob, trainer.TrainJobFailed); duration > 0 {
		durationLabels := r.withCondition(labels, trainer.TrainJobFailed)
		TrainJobDurationSeconds.With(durationLabels).Observe(duration)
	}

	// Decrement active count if the job was tracked as active
	if phase := r.getCurrentPhase(trainJob); r.isActivePhase(phase) {
		activeLabels := r.withPhase(labels, phase)
		TrainJobsActive.With(activeLabels).Dec()
	}
}

// RecordReconciliationDuration records the duration of a reconciliation operation.
func (r *MetricsRecorder) RecordReconciliationDuration(trainJob *trainer.TrainJob, duration time.Duration) {
	labels := r.getBaseLabels(trainJob)
	TrainJobReconciliationDuration.With(labels).Observe(duration.Seconds())
}

// RecordReconciliationError records an error during reconciliation.
func (r *MetricsRecorder) RecordReconciliationError(trainJob *trainer.TrainJob, errorType string) {
	labels := r.getBaseLabels(trainJob)
	labels["error_type"] = errorType
	TrainJobReconciliationErrors.With(labels).Inc()
}

// Helper methods

func (r *MetricsRecorder) getBaseLabels(trainJob *trainer.TrainJob) map[string]string {
	runtime := "unknown"
	if trainJob.Spec.RuntimeRef.Name != "" {
		runtime = trainJob.Spec.RuntimeRef.Name
	}

	return map[string]string{
		TrainJobNamespaceLabel: trainJob.Namespace,
		TrainJobRuntimeLabel:   runtime,
	}
}

func (r *MetricsRecorder) withPhase(labels map[string]string, phase string) map[string]string {
	newLabels := make(map[string]string, len(labels)+1)
	for k, v := range labels {
		newLabels[k] = v
	}
	newLabels[TrainJobPhaseLabel] = phase
	return newLabels
}

func (r *MetricsRecorder) withCondition(labels map[string]string, condition string) map[string]string {
	newLabels := make(map[string]string, len(labels)+1)
	for k, v := range labels {
		newLabels[k] = v
	}
	newLabels[TrainJobConditionLabel] = condition
	return newLabels
}

func (r *MetricsRecorder) getCurrentPhase(trainJob *trainer.TrainJob) string {
	if meta.IsStatusConditionTrue(trainJob.Status.Conditions, string(trainer.TrainJobSuspended)) {
		return "Suspended"
	}
	if meta.IsStatusConditionTrue(trainJob.Status.Conditions, string(trainer.TrainJobComplete)) {
		return "Complete"
	}
	if meta.IsStatusConditionTrue(trainJob.Status.Conditions, string(trainer.TrainJobFailed)) {
		return "Failed"
	}
	// If any jobs are running, consider the TrainJob as Running
	if len(trainJob.Status.JobsStatus) > 0 {
		return "Running"
	}
	return "Created"
}

func (r *MetricsRecorder) isActivePhase(phase string) bool {
	// Only track Created, Running, and Suspended as active phases
	// Complete and Failed are terminal states tracked separately
	return phase == "Created" || phase == "Running" || phase == "Suspended"
}

func (r *MetricsRecorder) calculateDuration(trainJob *trainer.TrainJob, conditionType string) float64 {
	if trainJob.CreationTimestamp.IsZero() {
		return 0
	}

	// Find the condition and use its LastTransitionTime
	for _, cond := range trainJob.Status.Conditions {
		if cond.Type == conditionType && cond.Status == metav1.ConditionTrue {
			if !cond.LastTransitionTime.IsZero() {
				return cond.LastTransitionTime.Sub(trainJob.CreationTimestamp.Time).Seconds()
			}
		}
	}

	return 0
}
