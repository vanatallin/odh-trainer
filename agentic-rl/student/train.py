#!/usr/bin/env python3
"""
Training script for Agentic RL with PPO

This script trains a language model agent using PPO and a reward model sidecar.
"""

import os
import logging
import time
from pathlib import Path

import torch

from agent import PPOAgent
from environment import TextGenerationEnvironment, PromptDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def wait_for_reward_model(url: str, timeout: int = 300, interval: int = 5):
    """
    Wait for the reward model to be ready.

    Args:
        url: URL of the reward model
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds
    """
    import requests

    logger.info(f"Waiting for reward model at {url}")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("Reward model is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.info(f"Reward model not ready yet, retrying in {interval}s...")
        time.sleep(interval)

    raise RuntimeError(f"Reward model at {url} did not become ready within {timeout}s")


def main():
    """Main training loop."""
    # Configuration from environment variables
    num_epochs = int(os.getenv('NUM_EPOCHS', '3'))
    batch_size = int(os.getenv('BATCH_SIZE', '4'))
    learning_rate = float(os.getenv('LEARNING_RATE', '1e-5'))
    ppo_epochs = int(os.getenv('PPO_EPOCHS', '4'))
    max_response_length = int(os.getenv('MAX_RESPONSE_LENGTH', '128'))
    model_name = os.getenv('MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    checkpoint_dir = os.getenv('CHECKPOINT_DIR', '/checkpoints')
    log_interval = int(os.getenv('LOG_INTERVAL', '10'))
    reward_model_url = os.getenv('REWARD_MODEL_URL', 'http://localhost:8080')

    logger.info("=" * 80)
    logger.info("Agentic RL Training with PPO")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"PPO epochs: {ppo_epochs}")
    logger.info(f"Max response length: {max_response_length}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Reward model URL: {reward_model_url}")
    logger.info("=" * 80)

    # Wait for reward model to be ready
    wait_for_reward_model(reward_model_url)

    # Initialize environment
    logger.info("Initializing environment...")
    env = TextGenerationEnvironment()
    prompt_dataset = PromptDataset(env.prompts, shuffle=True)

    # Initialize agent
    logger.info("Initializing PPO agent...")
    agent = PPOAgent(
        model_name=model_name,
        reward_model_url=reward_model_url,
        learning_rate=learning_rate,
        ppo_epochs=ppo_epochs,
    )

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    total_episodes = 0

    for epoch in range(num_epochs):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'=' * 80}")

        epoch_start_time = time.time()
        epoch_rewards = []

        # Iterate over prompts
        for batch_idx, prompts_batch in enumerate(
            [prompt_dataset.get_batch(batch_size) for _ in range(len(prompt_dataset) // batch_size)]
        ):
            batch_start_time = time.time()

            # Collect trajectories
            batch_prompts = []
            batch_responses = []
            batch_log_probs = []
            batch_rewards = []

            for prompt in prompts_batch:
                # Generate response
                response, log_probs, response_ids = agent.generate_response(
                    prompt,
                    max_length=max_response_length
                )

                # Get reward from reward model
                reward = agent.get_reward(prompt, response)

                # Store trajectory
                batch_prompts.append(prompt)
                batch_responses.append(response)
                batch_log_probs.append(log_probs)
                batch_rewards.append(reward)

                # Record in environment
                env.record_episode(prompt, response, reward)

                total_episodes += 1

            # Convert to tensors
            batch_rewards_tensor = torch.tensor(batch_rewards, device=agent.device)

            # Compute advantages
            advantages, returns = agent.compute_advantages(batch_rewards_tensor)

            # PPO update
            for ppo_epoch in range(ppo_epochs):
                total_policy_loss = 0
                total_kl = 0

                for i, prompt in enumerate(batch_prompts):
                    # Get new log probs
                    response_text = batch_responses[i]
                    old_log_probs = batch_log_probs[i]

                    # Tokenize for recomputation
                    inputs = agent.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(agent.device)

                    response_tokens = agent.tokenizer(
                        response_text,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).input_ids[0].to(agent.device)

                    full_ids = torch.cat([inputs.input_ids[0], response_tokens], dim=-1)
                    prompt_length = inputs.input_ids.shape[1]

                    # Forward pass
                    outputs = agent.model(full_ids.unsqueeze(0))
                    logits = outputs.logits[0, prompt_length - 1:-1]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(1, response_tokens.unsqueeze(-1)).squeeze(-1)

                    # Compute losses
                    policy_loss = agent.compute_policy_loss(
                        old_log_probs.detach(),
                        new_log_probs,
                        advantages[i].unsqueeze(0)
                    )

                    # Entropy bonus
                    probs = torch.exp(log_probs)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    entropy_loss = -agent.entropy_coef * entropy

                    # Total loss
                    loss = policy_loss + entropy_loss

                    # Backward pass
                    agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.max_grad_norm)
                    agent.optimizer.step()

                    total_policy_loss += policy_loss.item()

                    # Compute KL divergence
                    kl = agent.compute_kl_divergence(inputs.input_ids[0], response_tokens)
                    total_kl += kl

            # Logging
            batch_time = time.time() - batch_start_time
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            epoch_rewards.extend(batch_rewards)

            global_step += 1

            if global_step % log_interval == 0:
                stats = env.get_statistics()
                logger.info(
                    f"Step {global_step} | "
                    f"Batch {batch_idx + 1} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Policy Loss: {total_policy_loss / len(batch_prompts):.4f} | "
                    f"KL Div: {total_kl / len(batch_prompts):.4f} | "
                    f"Time: {batch_time:.2f}s"
                )
                logger.info(
                    f"Overall Stats - "
                    f"Episodes: {stats['total_episodes']} | "
                    f"Avg Reward: {stats['average_reward']:.4f} | "
                    f"Recent 100: {stats['recent_100_avg']:.4f}"
                )

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        epoch_avg_reward = sum(epoch_rewards) / len(epoch_rewards)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch + 1} Summary")
        logger.info(f"Average Reward: {epoch_avg_reward:.4f}")
        logger.info(f"Episodes: {len(epoch_rewards)}")
        logger.info(f"Time: {epoch_time:.2f}s")
        logger.info(f"{'=' * 80}\n")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch + 1}")
        agent.save_checkpoint(checkpoint_path)

    # Final statistics
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)

    stats = env.get_statistics()
    logger.info(f"Total Episodes: {stats['total_episodes']}")
    logger.info(f"Average Reward: {stats['average_reward']:.4f}")
    logger.info(f"Min Reward: {stats['min_reward']:.4f}")
    logger.info(f"Max Reward: {stats['max_reward']:.4f}")

    # Show best responses
    logger.info("\nTop 5 Best Responses:")
    for i, episode in enumerate(env.get_best_responses(top_k=5), 1):
        logger.info(f"\n{i}. Prompt: {episode.prompt}")
        logger.info(f"   Response: {episode.response[:100]}...")
        logger.info(f"   Reward: {episode.reward:.4f}")

    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final")
    agent.save_checkpoint(final_checkpoint_path)
    logger.info(f"\nFinal model saved to {final_checkpoint_path}")


if __name__ == '__main__':
    main()
