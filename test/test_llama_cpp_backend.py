from pathlib import Path
from unittest.mock import patch

from src.generation.llama_cpp_backend import LlamaCppGenerator


def test_llama_cpp_backend_extracts_answer_from_truncated_prompt_ui(tmp_path):
    executable = tmp_path / "llama-cli.exe"
    model = tmp_path / "model.gguf"
    executable.touch()
    model.touch()
    console = "banner\n> long prompt ... (truncated)\nGrounded answer [S1].\n[ Prompt: 20 t/s ]"

    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = console
        run.return_value.stderr = ""
        answer = LlamaCppGenerator(Path(executable), Path(model))("long prompt with omitted middle")

    assert answer == "Grounded answer [S1]."
    command = run.call_args.args[0]
    assert command[command.index("--prompt") + 1] == "long prompt with omitted middle"
    assert "--single-turn" in command
