"""Weight-Space Programming Gym (wsp_gym)."""

from wsp_gym.model import ModelConfig, forward_with_cache, init_weights
from wsp_gym.gauge import scramble_gauge
from wsp_gym.generators import generate_challenge, CATEGORIES
from wsp_gym.verifier import verify_submission
from wsp_gym.sandbox import SandboxSession, generate_suite

__all__ = [
    "ModelConfig",
    "init_weights",
    "forward_with_cache",
    "scramble_gauge",
    "generate_challenge",
    "CATEGORIES",
    "verify_submission",
    "SandboxSession",
    "generate_suite",
]
