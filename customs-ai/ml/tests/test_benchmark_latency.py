"""Latency benchmark yadrosi — sof/deterministik birliklar (onnxruntime kerak EMAS)."""
from __future__ import annotations

import numpy as np

from benchmark_latency import (
    DEFAULT_BUDGET_MS,
    _import_postprocess,
    bench_postprocess,
    bench_preprocess,
    check_budget,
    percentile,
    summarize,
    synth_image,
    synth_yolo_output,
)


# --- percentile (lineer interpolatsiya) ---
def test_percentile_edges_and_interp():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 4.0
    assert percentile(vals, 50) == 2.5  # (4-1)*0.5 = 1.5 -> 2.0/3.0 orasi
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 95) == 7.0


# --- summarize determinizmi ---
def test_summarize_deterministic_and_order_invariant():
    a = summarize([10, 20, 30, 40, 50])
    b = summarize([50, 40, 30, 20, 10])
    assert a == b  # tartibga bog'liq emas
    assert a["n"] == 5 and a["min"] == 10.0 and a["max"] == 50.0
    assert a["p50"] == 30.0


def test_summarize_empty():
    s = summarize([])
    assert s["n"] == 0 and s["p95"] == 0.0


# --- gate mantig'i ---
def test_check_budget_pass_fail():
    ok, _ = check_budget(250.0, DEFAULT_BUDGET_MS)
    assert ok
    bad, reason = check_budget(350.0, DEFAULT_BUDGET_MS)
    assert not bad and "buzildi" in reason
    # chegara: p95 == budjet -> PASS
    assert check_budget(300.0, 300.0)[0]


# --- sintetik kirish shakli ---
def test_synth_yolo_output_shape_and_determinism():
    a = synth_yolo_output(num_classes=7, num_preds=8400, imgsz=640, seed=0)
    b = synth_yolo_output(num_classes=7, num_preds=8400, imgsz=640, seed=0)
    assert a.shape == (1, 11, 8400)  # 4 + 7 class
    assert np.array_equal(a, b)  # seed deterministik


def test_synth_image_shape():
    img = synth_image(h=480, w=640, seed=1)
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8


# --- pre/post-process smoke (backend funksiyalari yuklanadi) ---
def test_pipeline_smoke_runs_and_is_fast():
    pp = _import_postprocess()
    assert pp is not None, "backend yolo_postprocess yuklanmadi"
    img = synth_image(h=480, w=640, seed=0)
    raw = synth_yolo_output(num_classes=7, imgsz=640, seed=0)

    pre = summarize(bench_preprocess(img, 640, iters=3, warmup=1, pp=pp))
    post = summarize(bench_postprocess(raw, 640, 640, 480, 0.20, 0.50,
                                       iters=3, warmup=1, pp=pp))
    assert pre["n"] == 3 and post["n"] == 3
    # Sof-numpy pre/post-process budjetning kichik ulushi bo'lishi kerak.
    assert pre["p95"] < DEFAULT_BUDGET_MS
    assert post["p95"] < DEFAULT_BUDGET_MS
