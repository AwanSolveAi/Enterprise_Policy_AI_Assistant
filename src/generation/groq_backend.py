"""Groq generation backend for Enterprise Policy AI.

Provides a simple prompt -> answer callable compatible with
app.service.load_generator().
"""

from __future__ import annotations

import os

from groq import Groq


MODEL_NAME = os.getenv("POLICY_GROQ_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = int(os.getenv("POLICY_GROQ_MAX_TOKENS", "1024"))
TIMEOUT_SECONDS = float(os.getenv("POLICY_GROQ_TIMEOUT", "60"))


def generate(prompt: str) -> str:
    """Generate one grounded answer from the supplied RAG prompt."""

    prompt = str(prompt).strip()

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in the current environment."
        )

    client = Groq(
        api_key=api_key,
        timeout=TIMEOUT_SECONDS,
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
    )

    if not response.choices:
        raise RuntimeError("Groq returned no completion choices.")

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty answer.")

    return answer.strip()