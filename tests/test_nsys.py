"""Run the trainer's capture boundaries on CPU with mocked CUDA calls."""

import ast
import inspect

import pytest

from modded_nanogpt_moe import model, train


def _capture_boundaries():
    # Execute the real guarded blocks, without initializing NCCL, compiling a
    # model, or loading training data. This catches omitted trainer wiring.
    tree = ast.parse(inspect.getsource(train.main))

    def has_profiler_call(node):
        return any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func) in (
                "torch.cuda.profiler.start", "torch.cuda.profiler.stop")
            for child in ast.walk(node)
        )

    boundaries = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and has_profiler_call(node)
        and not any(
            isinstance(child, ast.If) and has_profiler_call(child)
            for child in ast.walk(node) if child is not node
        )
    ]
    boundaries.sort(key=lambda node: node.lineno)
    assert len(boundaries) == 3  # Start, scheduled stop, early-stop cleanup.
    return [
        compile(ast.Module(body=[node], type_ignores=[]), train.__file__, "exec")
        for node in boundaries
    ]


@pytest.fixture
def capture_runtime(monkeypatch):
    events = []
    monkeypatch.setattr(model, "_moe_nsys_capture_active", False)
    monkeypatch.setattr(train.torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(train.torch.cuda.profiler, "start", lambda: events.append("start"))
    monkeypatch.setattr(train.torch.cuda.profiler, "stop", lambda: events.append("stop"))
    namespace = dict(vars(train))
    namespace.update(
        nsys_profile=True, nsys_capture_active=False,
        nsys_warmup_steps=10, nsys_active_steps=2,
        print0=lambda *args, **kwargs: None,
    )
    return namespace, events


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_enables_moe_ranges_only_for_captured_updates(capture_runtime, enabled):
    namespace, events = capture_runtime
    start, stop, _ = _capture_boundaries()
    namespace["nsys_profile"] = enabled
    captured_updates = []
    for step in range(14):
        namespace["step"] = step
        exec(start, namespace)
        assert model._moe_nsys_capture_active == namespace["nsys_capture_active"]
        if model._moe_nsys_capture_active:
            captured_updates.append(step + 1)
        exec(stop, namespace)
    assert captured_updates == ([11, 12] if enabled else [])
    assert events == (["sync", "start", "sync", "stop"] if enabled else [])
    assert not model._moe_nsys_capture_active
    assert not namespace["nsys_capture_active"]


@pytest.mark.parametrize("early_stop", [False, True])
@pytest.mark.parametrize("failure", [None, "synchronize", "stop"])
def test_trainer_stop_resets_moe_flag_even_on_failure(
        capture_runtime, monkeypatch, early_stop, failure):
    namespace, _ = capture_runtime
    start, stop, early_cleanup = _capture_boundaries()
    namespace["step"] = 10
    exec(start, namespace)
    assert model._moe_nsys_capture_active
    namespace.update(step=11, completed_updates=12)

    if failure is not None:
        def fail():
            raise RuntimeError("mock shutdown failure")

        target = train.torch.cuda if failure == "synchronize" else train.torch.cuda.profiler
        monkeypatch.setattr(target, failure, fail)
        with pytest.raises(RuntimeError, match="mock shutdown failure"):
            exec(early_cleanup if early_stop else stop, namespace)
    else:
        exec(early_cleanup if early_stop else stop, namespace)

    assert not model._moe_nsys_capture_active
    assert not namespace["nsys_capture_active"]
