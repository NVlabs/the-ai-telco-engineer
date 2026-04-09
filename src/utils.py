# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared utilities for the agent and agent manager."""

import random
import time

import printer


# Rate limit retry configuration
MAX_RATE_LIMIT_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10
MAX_BACKOFF_SECONDS = 120


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit error (HTTP 429)."""
    error_str = str(error).lower()
    return (
        "429" in error_str
        or "rate limit" in error_str
        or "rate_limit" in error_str
        or "too many requests" in error_str
        or "model capacity reached" in error_str
    )


def invoke_llm_with_retry(llm, prompt: str, context: str = "LLM"):
    """
    Invoke LLM with a prompt; retry with exponential backoff on rate limit errors.
    Returns the response object (use .content for the string).
    Raises the last exception after max retries or on non-rate-limit errors.
    """
    last_error = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e) and attempt < MAX_RATE_LIMIT_RETRIES:
                backoff = min(
                    INITIAL_BACKOFF_SECONDS * (2 ** attempt),
                    MAX_BACKOFF_SECONDS,
                )
                jitter = random.uniform(0, backoff * 0.25)
                wait_time = backoff + jitter
                printer.log(
                    f"Rate limit hit ({context}), attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}. "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
                continue
            raise last_error
