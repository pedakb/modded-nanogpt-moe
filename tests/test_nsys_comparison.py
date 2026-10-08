"""Synthetic timeline checks; no CUDA execution or Nsight collection."""
import sqlite3
import csv
import shutil
import sys

import pytest

from tools.compare_nsys_experts import analyze, main, union_duration, validate_capture


def test_union_does_not_double_count_overlapping_streams():
    assert union_duration([(0, 10), (3, 7), (8, 12), (15, 20)]) == 17


def test_launch_enclosure_attribution_and_gpu_tail(tmp_path, monkeypatch):
    path = tmp_path / "trace.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        INSERT INTO StringIds VALUES (1,'gemm'),(2,'cudaLaunchKernel'),(3,'cudaDeviceSynchronize');
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
        INSERT INTO NVTX_EVENTS VALUES
            (0,100,16777217,'optimizer_step.update_6',NULL),
            (100,200,16777217,'optimizer_step.update_7',NULL),
            (10,30,16777217,'grouped_gemm.fc1.forward',NULL),
            (110,130,16777217,'optimizer_update.index_1.Muon',NULL),
            (140,160,16777218,'moe.pack',NULL);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (15,20,16777217,1,2),(115,120,16777217,2,2),
            (145,150,16777217,3,2),(200,240,16777217,4,3);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (25,75,16777216,1,1),(125,175,16777216,2,1),(170,230,16777216,3,1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (80,90);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER);
    """)
    db.commit()
    db.close()
    metrics, top, missing = analyze(path)
    assert metrics["grouped_gemm_kernel_sum_ms_per_update"] == 50 / 2e6
    assert metrics["muon_kernel_sum_ms_per_update"] == 50 / 2e6
    # Same wall time but different CPU thread: must remain unattributed.
    assert metrics["router_dispatch_forward_kernel_sum_ms_per_update"] == 0
    assert metrics["unattributed_kernel_sum_ms_per_update"] == 60 / 2e6
    assert metrics["capture_envelope_ms_per_update"] == 230 / 2e6
    assert metrics["gpu_busy_union_ms_per_update"] == 165 / 2e6
    assert metrics["gpu_no_recorded_activity_ms_per_update"] == 65 / 2e6
    assert metrics["cuda_runtime_sync_sum_ms_per_update"] == 40 / 2e6
    assert top == [("gemm", [160, 3])]
    assert missing == []
    with pytest.raises(ValueError, match="expected 1"):
        analyze(path, expected_updates=1)
    for name in ("moe_e64k8", "moe_e256k6", "moe_e256k6_shared"):
        directory = tmp_path / name
        directory.mkdir()
        shutil.copyfile(path, directory / "profile.sqlite")
    monkeypatch.setattr(sys, "argv", ["compare_nsys_experts", str(tmp_path)])
    main()
    with (tmp_path / "comparison.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(float(row["delta_vs_e64_muon_kernel_sum_ms_per_update"]) == 0 for row in rows)
    assert "379 ms" in (tmp_path / "summary.md").read_text()
    # Add the production NVTX child ranges required for capture acceptance.
    with sqlite3.connect(path) as db:
        for base in (0, 100):
            for phase in ("forward", "backward"):
                for i in range(8):
                    db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                               (base+1+i, base+2+i, 16777217, f"{phase}.microbatch_{i}"))
            for name in ("data_preparation", "optimizer_update.index_0.AdamW"):
                db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                           (base+1, base+2, 16777217, name))
        db.execute("INSERT INTO NVTX_EVENTS VALUES (1,2,16777217,'optimizer_update.index_1.Muon',NULL)")
    log_path = tmp_path / "execution.log"
    success = "Capture range started in the application.\nCapture range ended in the application.\n"
    termination = "torch.distributed.elastic.multiprocessing.api.SignalException: Process 42 got signal: 15\n"
    log_path.write_text(success + termination)
    for status in (0, 1, 143):
        validate_capture(path, log_path, status)
    # Nsight buffers stdout: the real log prints capture notices after the
    # torchrun SIGTERM traceback (sometimes twice due to repeated SIGTERM).
    log_path.write_text(termination + termination + success)
    validate_capture(path, log_path, 1)
    log_path.write_text(success + termination)
    with pytest.raises(ValueError, match="exit 2"):
        validate_capture(path, log_path, 2)
    log_path.write_text(success)
    with pytest.raises(ValueError, match="verified expected"):
        validate_capture(path, log_path, 1)
    log_path.write_text(success + "RuntimeError: CUDA failure\n" + termination)
    with pytest.raises(ValueError, match="Unexpected error"):
        validate_capture(path, log_path, 1)
    log_path.write_text("Capture range started in the application.\n" + termination)
    with pytest.raises(ValueError, match="capture start"):
        validate_capture(path, log_path, 1)
    log_path.write_text(success + termination)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE NVTX_EVENTS SET text='optimizer_step.update_8' WHERE text='optimizer_step.update_7'")
    with pytest.raises(ValueError, match="exactly updates 6 and 7"):
        validate_capture(path, log_path, 1)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE NVTX_EVENTS SET text='optimizer_step.update_7',end=NULL WHERE text='optimizer_step.update_8'")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_capture(path, log_path, 1)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE NVTX_EVENTS SET end=200 WHERE text='optimizer_step.update_7'")
        db.execute("DELETE FROM NVTX_EVENTS WHERE text='backward.microbatch_7' AND start > 100")
    with pytest.raises(ValueError, match="microbatch"):
        validate_capture(path, log_path, 1)
