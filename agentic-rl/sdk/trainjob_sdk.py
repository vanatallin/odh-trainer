#!/usr/bin/env python3
"""
Agentic RL Training SDK with TrainingRuntime Support

This script extends the Kubeflow Training SDK by adding TrainingRuntime
creation capabilities for agentic RL with sidecar reward models.

Uses upstream Kubeflow Training SDK: https://www.kubeflow.org/docs/components/trainer/getting-started/
"""

from typing import Dict
from kubeflow.trainer import TrainerClient, CustomTrainer
import yaml
import subprocess
import tempfile
import os


class AgenticRLTrainingSDK:
    """
    Extended SDK for agentic RL training jobs.

    Provides both TrainingRuntime creation (YAML-based) and TrainJob
    submission (using upstream Kubeflow Training SDK).
    """

    def __init__(
        self,
        namespace: str = "default",
        student_image: str = "your-registry/student-agent:latest",
        reward_model_image: str = "your-registry/reward-model:latest",
    ):
        """
        Initialize the SDK.

        Args:
            namespace: Kubernetes namespace
            student_image: Container image for student agent
            reward_model_image: Container image for reward model sidecar
        """
        self.namespace = namespace
        self.student_image = student_image
        self.reward_model_image = reward_model_image
        self.client = TrainerClient(namespace=namespace)

    def create_training_runtime(
        self,
        runtime_name: str = "agentic-rl-pytorch",
        cluster_scoped: bool = False,
    ) -> Dict:
        """
        Create a TrainingRuntime with sidecar configuration.

        Args:
            runtime_name: Name for the TrainingRuntime
            cluster_scoped: Whether to create ClusterTrainingRuntime

        Returns:
            TrainingRuntime specification dict
        """
        runtime_spec = {
            "apiVersion": "trainer.kubeflow.org/v1alpha1",
            "kind": "ClusterTrainingRuntime" if cluster_scoped else "TrainingRuntime",
            "metadata": {
                "name": runtime_name,
            },
            "spec": {
                "mlPolicy": {
                    "numNodes": 1,
                    "torch": {
                        "numProcPerNode": "auto"
                    }
                },
                "template": {
                    "spec": {
                        "replicatedJobs": [
                            {
                                "name": "node",
                                "template": {
                                    "metadata": {
                                        "labels": {
                                            "trainer.kubeflow.org/trainjob-ancestor-step": "trainer"
                                        }
                                    },
                                    "spec": {
                                        "backoffLimit": 2,
                                        "template": {
                                            "spec": {
                                                "containers": [
                                                    # Student container
                                                    {
                                                        "name": "node",
                                                        "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
                                                        "command": ["python", "/workspace/train.py"],
                                                        "env": [
                                                            {"name": "REWARD_MODEL_URL", "value": "http://localhost:8080"},
                                                            {"name": "PYTHONUNBUFFERED", "value": "1"},
                                                        ],
                                                        "resources": {
                                                            "requests": {
                                                                "nvidia.com/gpu": "1",
                                                                "memory": "8Gi",
                                                                "cpu": "4"
                                                            },
                                                            "limits": {
                                                                "nvidia.com/gpu": "1",
                                                                "memory": "8Gi"
                                                            }
                                                        },
                                                        "volumeMounts": [
                                                            {"name": "workspace", "mountPath": "/workspace"},
                                                            {"name": "checkpoints", "mountPath": "/checkpoints"},
                                                            {"name": "dshm", "mountPath": "/dev/shm"}
                                                        ]
                                                    },
                                                    # Reward model sidecar
                                                    {
                                                        "name": "reward-model",
                                                        "image": "python:3.10-slim",
                                                        "command": ["python", "/app/server.py"],
                                                        "env": [
                                                            {"name": "PORT", "value": "8080"},
                                                            {"name": "MODEL_NAME", "value": "distilbert-base-uncased"},
                                                            {"name": "DEVICE", "value": "cuda"},
                                                            {"name": "PYTHONUNBUFFERED", "value": "1"},
                                                        ],
                                                        "ports": [
                                                            {"containerPort": 8080, "name": "http", "protocol": "TCP"}
                                                        ],
                                                        "resources": {
                                                            "requests": {
                                                                "nvidia.com/gpu": "1",
                                                                "memory": "4Gi",
                                                                "cpu": "2"
                                                            },
                                                            "limits": {
                                                                "nvidia.com/gpu": "1",
                                                                "memory": "4Gi"
                                                            }
                                                        },
                                                        "volumeMounts": [
                                                            {"name": "checkpoints", "mountPath": "/checkpoints", "readOnly": True},
                                                            {"name": "dshm", "mountPath": "/dev/shm"}
                                                        ],
                                                        "readinessProbe": {
                                                            "httpGet": {"path": "/health", "port": 8080},
                                                            "initialDelaySeconds": 30,
                                                            "periodSeconds": 10,
                                                            "timeoutSeconds": 5,
                                                            "failureThreshold": 3
                                                        },
                                                        "livenessProbe": {
                                                            "httpGet": {"path": "/health", "port": 8080},
                                                            "initialDelaySeconds": 60,
                                                            "periodSeconds": 30,
                                                            "timeoutSeconds": 5,
                                                            "failureThreshold": 3
                                                        }
                                                    }
                                                ],
                                                "volumes": [
                                                    {"name": "workspace", "emptyDir": {}},
                                                    {"name": "checkpoints", "emptyDir": {}},
                                                    {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}}
                                                ],
                                                "restartPolicy": "Never"
                                            }
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }

        if not cluster_scoped:
            runtime_spec["metadata"]["namespace"] = self.namespace

        return runtime_spec

    def deploy_runtime(self, runtime_spec: Dict, dry_run: bool = False):
        """
        Deploy a TrainingRuntime using kubectl.

        Args:
            runtime_spec: TrainingRuntime specification
            dry_run: If True, only validate without applying
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(runtime_spec, f)
            temp_file = f.name

        try:
            cmd = ["kubectl", "apply", "-f", temp_file]
            if dry_run:
                cmd.append("--dry-run=client")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ Successfully applied {runtime_spec['kind']}/{runtime_spec['metadata']['name']}")
                print(result.stdout)
            else:
                print(f"✗ Failed to apply {runtime_spec['kind']}/{runtime_spec['metadata']['name']}")
                print(result.stderr)
                raise RuntimeError(f"kubectl apply failed: {result.stderr}")

        finally:
            os.unlink(temp_file)

    def create_train_job(
        self,
        name: str,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        ppo_epochs: int = 4,
        num_nodes: int = 1,
        gpu_per_node: int = 1,
        memory_per_node: str = "8Gi",
        cpu_per_node: int = 4,
    ) -> str:
        """
        Create and submit a TrainJob using Kubeflow Training SDK.

        Args:
            name: Name of the TrainJob
            model_name: Language model to use
            num_epochs: Number of training epochs
            batch_size: Batch size for PPO
            learning_rate: Learning rate
            ppo_epochs: PPO optimization epochs
            num_nodes: Number of training nodes
            gpu_per_node: GPUs per node
            memory_per_node: Memory per node
            cpu_per_node: CPUs per node

        Returns:
            Job ID (name) of the created job
        """

        def train_agentic_rl():
            """Training function executed in the training pod."""
            import os
            import sys

            # Import training modules
            sys.path.insert(0, '/workspace')
            from student.agent import PPOAgent
            from student.environment import TextGenerationEnvironment

            # Configuration
            model = os.getenv('MODEL_NAME', model_name)
            epochs = int(os.getenv('NUM_EPOCHS', str(num_epochs)))
            batch = int(os.getenv('BATCH_SIZE', str(batch_size)))
            lr = float(os.getenv('LEARNING_RATE', str(learning_rate)))

            print(f"Starting agentic RL training with {model}")

            # Initialize
            agent = PPOAgent(model_name=model, learning_rate=lr, ppo_epochs=ppo_epochs)
            env = TextGenerationEnvironment()

            # Training loop
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                trajectories = env.collect_trajectories(agent, batch_size=batch)
                rewards = env.get_rewards(trajectories)
                loss = agent.update(trajectories, rewards)
                print(f"Avg reward: {sum(rewards)/len(rewards):.4f}, Loss: {loss:.4f}")

            print("\nTraining complete!")

        # Submit using Kubeflow Training SDK
        from kubeflow.trainer import CustomTrainer

        job_id = self.client.train(
            name=name,
            trainer=CustomTrainer(
                func=train_agentic_rl,
                num_nodes=num_nodes,
                resources_per_node={
                    "cpu": cpu_per_node,
                    "memory": memory_per_node,
                    "gpu": gpu_per_node,
                },
            ),
        )

        return job_id

    def get_job(self, name: str):
        """Get job details."""
        return self.client.get_job(name=name)

    def get_job_logs(self, name: str, follow: bool = False):
        """Get job logs."""
        return self.client.get_job_logs(name=name, follow=follow)

    def list_jobs(self):
        """List all training jobs."""
        return self.client.list_jobs()

    def delete_job(self, name: str):
        """Delete a training job."""
        return self.client.delete_job(name=name)


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy agentic RL training")
    parser.add_argument("--mode", choices=["simple", "distributed"], default="simple",
                       help="Training mode")
    parser.add_argument("--namespace", default="default",
                       help="Kubernetes namespace")
    parser.add_argument("--student-image", required=True,
                       help="Student agent container image")
    parser.add_argument("--reward-image", required=True,
                       help="Reward model container image")
    parser.add_argument("--create-runtime", action="store_true",
                       help="Create TrainingRuntime before job")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate without applying")
    parser.add_argument("--follow-logs", action="store_true",
                       help="Follow logs after submission")

    args = parser.parse_args()

    # Create SDK client
    sdk = AgenticRLTrainingSDK(
        namespace=args.namespace,
        student_image=args.student_image,
        reward_model_image=args.reward_image,
    )

    # Create runtime if requested
    if args.create_runtime:
        print("Creating TrainingRuntime...")
        runtime = sdk.create_training_runtime("agentic-rl-pytorch")
        sdk.deploy_runtime(runtime, dry_run=args.dry_run)
        print()

    if not args.dry_run:
        # Create and submit job using Kubeflow Training SDK
        if args.mode == "simple":
            job_name = "agentic-rl-simple"
            print(f"Creating TrainJob '{job_name}'...")
            job_id = sdk.create_train_job(
                name=job_name,
                num_epochs=3,
                batch_size=4,
                learning_rate=1e-5,
            )
        else:
            job_name = "agentic-rl-distributed"
            print(f"Creating TrainJob '{job_name}'...")
            job_id = sdk.create_train_job(
                name=job_name,
                num_epochs=10,
                batch_size=8,
                learning_rate=2e-5,
                num_nodes=4,
                gpu_per_node=2,
                memory_per_node="16Gi",
                cpu_per_node=8,
            )

        print(f"✓ TrainJob '{job_id}' created successfully!")

        # Show job status
        job = sdk.get_job(job_id)
        print(f"\nJob Status: {job.status}")

        # Follow logs if requested
        if args.follow_logs:
            print("\nFollowing logs (Ctrl+C to stop)...")
            try:
                for logline in sdk.get_job_logs(job_id, follow=True):
                    print(logline)
            except KeyboardInterrupt:
                print("\nStopped following logs")

    print("\nDone!")


if __name__ == "__main__":
    main()
