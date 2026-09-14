"""Minimal llama.cpp command-line generation backend.

This adapter intentionally implements the same ``prompt -> text`` callable
contract as the existing generators and does not know anything about retrieval,
prompt construction, or answer guardrails.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LlamaCppGenerator:
    executable: Path
    model: Path
    threads: int = 4
    context_size: int = 8192
    max_tokens: int = 120
    temperature: float = 0.0
    timeout_seconds: float = 300.0

    def __call__(self, prompt: str) -> str:
        if not self.executable.is_file():
            raise FileNotFoundError(f"llama.cpp executable not found: {self.executable}")
        if not self.model.is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.model}")

        command = [
            str(self.executable),
            "--model",
            str(self.model),
            "--prompt",
            prompt,
            "--threads",
            str(self.threads),
            "--ctx-size",
            str(self.context_size),
            "--n-predict",
            str(self.max_tokens),
            "--temp",
            str(self.temperature),
            "--seed",
            "0",
            "--no-display-prompt",
            "--simple-io",
            "--single-turn",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"llama.cpp exited with code {completed.returncode}: {detail}")
        output = completed.stdout
        # llama-cli's Windows chat UI echoes the prompt even when
        # --no-display-prompt is set. Keep that transport noise out of the
        # provider-neutral generator contract.
        if prompt in output:
            output = output.rsplit(prompt, 1)[1]
        elif "... (truncated)" in output:
            output = output.rsplit("... (truncated)", 1)[1]
        output = output.split("[ Prompt:", 1)[0].strip()
        if not output:
            raise RuntimeError("llama.cpp returned an empty response")
        return output
