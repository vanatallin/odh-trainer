#!/usr/bin/env python3
"""
Reward Model Sidecar Server

This server exposes a REST API for scoring text responses.
It loads a pre-trained reward model and returns scores for prompt-response pairs.
"""

import os
import json
import logging
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RewardModel:
    """Wrapper for the reward model."""

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the reward model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run on (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reward model: {model_name} on {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,  # Regression task for reward scoring
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Reward model loaded successfully")

    def score(self, prompt: str, response: str) -> float:
        """
        Score a prompt-response pair.

        Args:
            prompt: The input prompt
            response: The generated response

        Returns:
            Reward score (0-1)
        """
        # Combine prompt and response
        text = f"{prompt}\n\nResponse: {response}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        # Get score
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).item()

        return score


class RewardModelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the reward model API."""

    # Shared reward model instance
    reward_model = None
    model_lock = Lock()

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        """Set response headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def _send_json(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response."""
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data).encode())

    def _send_error_json(self, message: str, status_code: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status_code)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self._send_json({
                "status": "healthy",
                "model_loaded": self.reward_model is not None,
                "device": self.reward_model.device if self.reward_model else "unknown"
            })
        elif self.path == '/':
            self._send_json({
                "service": "Reward Model Server",
                "version": "1.0.0",
                "endpoints": {
                    "GET /health": "Health check",
                    "POST /score": "Score a prompt-response pair"
                }
            })
        else:
            self._send_error_json("Endpoint not found", 404)

    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/score':
            try:
                # Read request body
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                data = json.loads(body.decode())

                # Validate input
                if 'prompt' not in data or 'response' not in data:
                    self._send_error_json(
                        "Missing required fields: 'prompt' and 'response'",
                        400
                    )
                    return

                prompt = data['prompt']
                response = data['response']

                # Score the response
                with self.model_lock:
                    score = self.reward_model.score(prompt, response)

                # Return the score
                self._send_json({
                    "score": score,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "response_length": len(response)
                })

                logger.info(f"Scored response: {score:.4f}")

            except json.JSONDecodeError:
                self._send_error_json("Invalid JSON in request body", 400)
            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                self._send_error_json(str(e), 500)
        else:
            self._send_error_json("Endpoint not found", 404)

    def log_message(self, format, *args):
        """Override to use logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")


def main():
    """Main entry point."""
    # Configuration from environment
    port = int(os.getenv('PORT', '8080'))
    model_name = os.getenv('MODEL_NAME', 'distilbert-base-uncased')
    device = os.getenv('DEVICE', 'cuda')
    model_path = os.getenv('REWARD_MODEL_PATH', None)

    # Use custom model path if provided, otherwise use HuggingFace model
    model_to_load = model_path if model_path and os.path.exists(model_path) else model_name

    logger.info(f"Starting Reward Model Server on port {port}")
    logger.info(f"Model: {model_to_load}")
    logger.info(f"Device: {device}")

    try:
        # Initialize the reward model
        RewardModelHandler.reward_model = RewardModel(model_to_load, device)

        # Start the server
        server = HTTPServer(('0.0.0.0', port), RewardModelHandler)
        logger.info(f"Server listening on 0.0.0.0:{port}")
        logger.info("Ready to accept requests")

        server.serve_forever()

    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.shutdown()
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
