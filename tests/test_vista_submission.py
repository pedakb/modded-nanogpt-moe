"""CPU-only submission argument tests; never invoke a real scheduler."""
import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/vista/train.sh"


@pytest.mark.parametrize("options,expected", [
    ([], []),
    (["--job-name", "comparison suite"], ["--job-name=comparison suite"]),
    (["--account", "allocation"], ["--account=allocation"]),
    (["--sbatch-arg=--partition=gh", "--sbatch-arg", "--exclusive"],
     ["--partition=gh", "--exclusive"]),
    (["--job-name", "suite", "--account", "allocation", "--time", "08:00:00",
      "--sbatch-arg", "--comment=literal $(printf injected)"],
     ["--time=08:00:00", "--job-name=suite", "--account=allocation",
      "--comment=literal $(printf injected)"]),
])
def test_submission_parser_and_forwarding(tmp_path, options, expected):
    source = LAUNCHER.read_text()
    # Run the actual parser and sbatch block without the unrelated Bash-4-only
    # config preflight. This also works with macOS's bundled Bash 3.2.
    parser = source[source.index("submit=0\n"):source.index("# Resolve every config")]
    start = source.index('if [[ "$submit" -eq 1 ]]; then\n')
    submit = source[start:source.index('if [[ "$worker" -eq 1 &&', start)]
    capture = tmp_path / "args"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nexit 23\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
               CAPTURE=str(capture), LAUNCHER_PATH=str(LAUNCHER))
    env.pop("SLURM_MAIL_USER", None)
    script = ('set -euo pipefail\nusage() { :; }\n' + parser
              + '\nscript_path="$LAUNCHER_PATH"\nconfig_paths=("${config_arguments[@]}")\n'
              + submit)
    result = subprocess.run(
        ["bash", "-c", script, "test", "--submit", *options,
         "--checkpoint-interval", "100", "--resume", "/checkpoints/with spaces/latest.pt",
         "--", "configs/dense_baseline.toml", "configs/moe_e8k2_r2.toml"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    assert capture.read_text().splitlines() == [
        *expected, str(LAUNCHER), "--worker", "--checkpoint-interval", "100",
        "--resume", "/checkpoints/with spaces/latest.pt", "--",
        "configs/dense_baseline.toml", "configs/moe_e8k2_r2.toml",
    ]


@pytest.mark.parametrize("args,message", [
    (["--job-name"], "requires a nonempty value"),
    (["--account", ""], "requires a nonempty value"),
    (["--job-name", "--submit"], "requires a value"),
    (["--account", "--submit"], "requires a value"),
    (["--sbatch-arg"], "requires a nonempty value"),
    (["--job-name", "suite"], "require --submit"),
    (["--account", "allocation"], "require --submit"),
    (["--sbatch-arg", "--exclusive"], "require --submit"),
    (["--submit", "--sbatch-arg="], "expected one --option"),
    (["--submit", "--sbatch-arg", "script.sh"], "expected one --option"),
    (["--submit", "--sbatch-arg", "--"], "expected one --option"),
    (["--submit", "--sbatch-arg", "--partition gh"], "expected one --option"),
    (["--submit", "--sbatch-arg", "--wrap=echo hello"], "no --wrap"),
])
def test_scheduler_options_rejected_before_submission(args, message):
    # Every case fails before config preflight, module loading, or sbatch.
    result = subprocess.run(["bash", str(LAUNCHER), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert message in result.stderr
