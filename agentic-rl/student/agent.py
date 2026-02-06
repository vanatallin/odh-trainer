"""
RL Agent for Text Generation

This module implements the PPO agent for fine-tuning small language models.
Default model: TinyLlama-1.1B (1.1B parameters, efficient for RL training)
"""

import logging
from typing import List, Dict, Tuple, Optional
import requests
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent for text generation.
    """

    def __init__(
        self,
        model_name: str,
        reward_model_url: str,
        device: str = "cuda",
        learning_rate: float = 1e-5,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize the PPO agent.

        Args:
            model_name: HuggingFace model name
            reward_model_url: URL of the reward model service
            device: Device to run on
            learning_rate: Learning rate for optimization
            ppo_epochs: Number of PPO optimization epochs
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.reward_model_url = reward_model_url
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        logger.info(f"Loading model: {model_name} on {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Create reference model (frozen copy for KL divergence)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        logger.info("Agent initialized successfully")

    def generate_response(
        self,
        prompt: str,
        max_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate a response to a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum length of response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Tuple of (response_text, log_probs, values)
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        prompt_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=prompt_length + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0]
        response_ids = generated_ids[prompt_length:]

        # Decode response
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Calculate log probabilities
        with torch.no_grad():
            model_outputs = self.model(generated_ids.unsqueeze(0))
            logits = model_outputs.logits[0, prompt_length - 1:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, response_ids.unsqueeze(-1)).squeeze(-1)

        return response, action_log_probs, response_ids

    def get_reward(self, prompt: str, response: str, max_retries: int = 3) -> float:
        """
        Query the reward model for a score.

        Args:
            prompt: The prompt
            response: The generated response
            max_retries: Maximum number of retries

        Returns:
            Reward score
        """
        for attempt in range(max_retries):
            try:
                response_data = requests.post(
                    f"{self.reward_model_url}/score",
                    json={"prompt": prompt, "response": response},
                    timeout=10
                )
                response_data.raise_for_status()
                reward = response_data.json()["score"]
                return reward

            except requests.exceptions.RequestException as e:
                logger.warning(f"Reward model request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Failed to get reward after max retries, using default reward")
                    return 0.5  # Default neutral reward

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).

        Args:
            rewards: Tensor of rewards
            gamma: Discount factor
            lam: GAE lambda parameter

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []

        advantage = 0
        return_ = 0

        for reward in reversed(rewards):
            return_ = reward + gamma * return_
            returns.insert(0, return_)

            # Simplified advantage (no value function for now)
            advantage = reward + gamma * lam * advantage
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_policy_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PPO policy loss.

        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy
            advantages: Advantage estimates

        Returns:
            Policy loss
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        return policy_loss

    def compute_kl_divergence(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> float:
        """
        Compute KL divergence between current policy and reference policy.

        Args:
            prompt_ids: Tokenized prompt
            response_ids: Tokenized response

        Returns:
            KL divergence
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=-1).unsqueeze(0)

        with torch.no_grad():
            ref_outputs = self.ref_model(full_ids)
            ref_logits = ref_outputs.logits[0, len(prompt_ids) - 1:-1]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        curr_outputs = self.model(full_ids)
        curr_logits = curr_outputs.logits[0, len(prompt_ids) - 1:-1]
        curr_log_probs = F.log_softmax(curr_logits, dim=-1)

        kl = F.kl_div(curr_log_probs, ref_log_probs, reduction='batchmean', log_target=True)
        return kl.item()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Checkpoint loaded from {path}")
