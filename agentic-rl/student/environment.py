"""
Text Generation Environment for Agentic RL

This module defines the environment for training a language model agent
using reinforcement learning.
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Episode:
    """Represents a single episode (prompt-response cycle)."""
    prompt: str
    response: str
    reward: float


class TextGenerationEnvironment:
    """
    Environment for training text generation agents.

    The environment provides prompts and evaluates responses using
    a reward model.
    """

    def __init__(self, prompts: List[str] = None):
        """
        Initialize the environment.

        Args:
            prompts: List of prompts to use. If None, use default prompts.
        """
        self.prompts = prompts or self._get_default_prompts()
        self.current_prompt = None
        self.episode_history: List[Episode] = []

    def _get_default_prompts(self) -> List[str]:
        """Get default training prompts."""
        return [
            # Helpful assistant prompts
            "How do I make chocolate chip cookies?",
            "Explain the water cycle to a 10-year-old.",
            "What are the benefits of exercise?",
            "How can I improve my public speaking skills?",
            "Write a short poem about the ocean.",

            # Instruction following
            "Summarize the key points of healthy eating in 3 sentences.",
            "List 5 tips for better time management.",
            "Describe the process of photosynthesis.",
            "Explain what causes seasons on Earth.",
            "Give advice on how to start learning a new language.",

            # Creative tasks
            "Write a creative opening sentence for a mystery novel.",
            "Suggest 3 fun activities for a rainy day.",
            "Describe a peaceful forest scene.",
            "Write a haiku about technology.",
            "Create a short dialogue between two friends meeting after years.",

            # Problem solving
            "How would you troubleshoot a computer that won't turn on?",
            "What steps should I take to prepare for a job interview?",
            "How can I organize my workspace for better productivity?",
            "What are some strategies for managing stress?",
            "How do I choose a good book to read?",

            # Educational
            "What is the difference between weather and climate?",
            "Explain how a battery works.",
            "What are the primary colors and why?",
            "Describe the structure of an atom.",
            "How does the internet work in simple terms?",

            # Reasoning
            "Why is it important to recycle?",
            "What makes a good team leader?",
            "Why do we need sleep?",
            "What are the advantages of reading books?",
            "How does music affect our mood?",
        ]

    def reset(self) -> str:
        """
        Reset the environment and return a new prompt.

        Returns:
            A prompt string
        """
        self.current_prompt = random.choice(self.prompts)
        return self.current_prompt

    def get_batch_prompts(self, batch_size: int) -> List[str]:
        """
        Get a batch of prompts for parallel generation.

        Args:
            batch_size: Number of prompts to return

        Returns:
            List of prompt strings
        """
        return random.choices(self.prompts, k=batch_size)

    def record_episode(self, prompt: str, response: str, reward: float):
        """
        Record an episode in the history.

        Args:
            prompt: The prompt used
            response: The generated response
            reward: The reward received
        """
        episode = Episode(prompt=prompt, response=response, reward=reward)
        self.episode_history.append(episode)

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the training episodes.

        Returns:
            Dictionary with statistics
        """
        if not self.episode_history:
            return {
                "total_episodes": 0,
                "average_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
            }

        rewards = [ep.reward for ep in self.episode_history]
        return {
            "total_episodes": len(self.episode_history),
            "average_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "recent_100_avg": sum(rewards[-100:]) / min(100, len(rewards)),
        }

    def get_best_responses(self, top_k: int = 5) -> List[Episode]:
        """
        Get the best responses by reward.

        Args:
            top_k: Number of top responses to return

        Returns:
            List of top episodes
        """
        sorted_episodes = sorted(
            self.episode_history,
            key=lambda ep: ep.reward,
            reverse=True
        )
        return sorted_episodes[:top_k]


class PromptDataset:
    """
    Dataset for iterating over prompts during training.
    """

    def __init__(self, prompts: List[str], shuffle: bool = True):
        """
        Initialize the dataset.

        Args:
            prompts: List of prompts
            shuffle: Whether to shuffle prompts
        """
        self.prompts = prompts
        self.shuffle = shuffle
        self.current_index = 0

        if self.shuffle:
            random.shuffle(self.prompts)

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.prompts)
        return self

    def __next__(self) -> str:
        if self.current_index >= len(self.prompts):
            raise StopIteration

        prompt = self.prompts[self.current_index]
        self.current_index += 1
        return prompt

    def get_batch(self, batch_size: int) -> List[str]:
        """
        Get a batch of prompts.

        Args:
            batch_size: Size of the batch

        Returns:
            List of prompts
        """
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(self))
            except StopIteration:
                # Reset if we've exhausted the dataset
                self.__iter__()
                if len(batch) == 0:
                    batch.append(next(self))
                break
        return batch
