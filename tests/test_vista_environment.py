import os
import subprocess
from pathlib import Path

import pytest


ENV_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/vista/env.sh"


@pytest.mark.parametrize("sourced", [False, True])
@pytest.mark.parametrize("module_available", [False, True])
def test_vista_environment_fails_clearly_without_working_modules(
        tmp_path, sourced, module_available):
    if module_available:
        module = tmp_path / "module"
        module.write_text("#!/bin/bash\nexit 7\n")
        module.chmod(0o755)
    # No inherited module functions or shell startup files can mask the failure.
    environment = {
        "PATH": str(tmp_path),
        "HOME": str(tmp_path / "home"),
        "STOCKYARD": str(tmp_path / "stockyard"),
    }
    command = ["/bin/bash", "--noprofile", "--norc"]
    if sourced:
        command += [
            "-c", 'source "$1"; status=$?; printf "source_status=%s\\n" "$status"; exit "$status"',
            "bash", str(ENV_SCRIPT),
        ]
    else:
        command.append(str(ENV_SCRIPT))

    result = subprocess.run(command, env=environment, capture_output=True, text=True)

    assert result.returncode == 1
    assert result.stdout == ("source_status=1\n" if sourced else "")
    if module_available:
        assert result.stderr == (
            "Error: scripts/vista/env.sh could not load the Vista NVIDIA/CUDA modules.\n")
    else:
        assert result.stderr == (
            "Error: scripts/vista/env.sh requires the TACC module environment.\n")


@pytest.mark.parametrize("local_bin_present", [False, True])
@pytest.mark.parametrize("backend", [None, "extension"])
def test_vista_environment_can_be_sourced_twice_without_changing_run_state(
        tmp_path, local_bin_present, backend):
    home = tmp_path / "home [test]"
    local_bin = str(home / ".local/bin")
    initial_path = f"{local_bin}:/usr/bin:/bin" if local_bin_present else "/usr/bin:/bin"
    environment = {
        "PATH": initial_path,
        "HOME": str(home),
        "STOCKYARD": str(tmp_path / "stockyard"),
        "LD_LIBRARY_PATH": "/existing/cuda/lib:/existing/compiler/lib",
        "CC": "/stale/cc",
        "CXX": "/stale/cxx",
        "CUDAHOSTCXX": "/stale/cuda-host-cxx",
    }
    if backend is not None:
        environment["MOE_GMM_IMPLEMENTATION"] = backend
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", '''
set -euo pipefail
module() { printf 'module %s\n' "$*"; }
source "$1"
first_path="$PATH"
source "$1"
[[ "$PATH" == "$first_path" ]]
/usr/bin/env -0
''', "bash", str(ENV_SCRIPT)],
        env=environment, capture_output=True, text=True, check=True,
    )

    first_load, second_load, exported = result.stdout.split("\n", 2)
    assert first_load == second_load == "module load nvidia/25.3 cuda/12.9"
    resolved = dict(item.split("=", 1) for item in exported.split("\0") if item)
    expected_path = initial_path if local_bin_present else f"{initial_path}:{local_bin}"
    assert resolved["PATH"] == expected_path
    assert resolved["PATH"].split(os.pathsep).count(local_bin) == 1
    assert resolved["LD_LIBRARY_PATH"] == environment["LD_LIBRARY_PATH"]
    assert resolved["CC"] == "/usr/bin/gcc"
    assert resolved["CXX"] == resolved["CUDAHOSTCXX"] == "/usr/bin/g++"
    assert resolved["STOCKYARD"] == environment["STOCKYARD"]
    assert resolved["TB_ROOT"] == str(tmp_path / "stockyard/tensorboard")
    assert resolved["TB_SYSTEM"] == "vista"
    assert resolved.get("MOE_GMM_IMPLEMENTATION") == backend
