#!/usr/bin/env python3
"""Detection latency benchmark — REAL-TIME kritik yo'lni o'lchaydi (qabul mezoni #1).

Qabul mezoni: real X-ray -> detection < 300ms/rasm. "O'lchamasdan optimallashtirib
bo'lmaydi" — bu harness shu budjetni p50/p95 bilan tekshiradi va bosqichlarga
ajratadi (preprocess / inference / postprocess), shunda bottleneck ko'rinadi.

Rejimlar (graceful degrade):
  • TO'LIQ — `--model model.onnx` mavjud VA onnxruntime o'rnatilgan bo'lsa:
    haqiqiy `OnnxYoloDetector.detect()` uchidan-uchiga o'lchanadi (ADR-002
    provider tartibi bilan).
  • NUMPY-ONLY — model/onnxruntime bo'lmasa: sof-numpy pre/post-process (men
    egalik qiladigan, budjetning model'dan MUSTAQIL qismi) sintetik kirishda
    o'lchanadi; inference "o'lchanmadi" deb belgilanadi. Bu CI'da modelsiz ham
    regressiyani ushlaydi.

Yadro (`summarize`, `percentile`, `check_budget`, `synth_*`) sof Python/numpy —
onnxruntime kerak EMAS, shuning uchun deterministik birlik testi yoziladi.

Foydalanish:
    python ml/benchmark_latency.py --model backend/model_registry/v0_baseline/model.onnx \
        --image sample_xray.png --iters 50 --warmup 5 --budget-ms 300
    python ml/benchmark_latency.py            # modelsiz: numpy-only smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Qabul mezoni: real-time kritik yo'l. p95 shu budjetdan oshsa gate FAIL.
DEFAULT_BUDGET_MS = 300.0


# ---------------------------------------------------------------------------
# Sof yadro (deterministik, og'ir deps'siz — testlardan to'g'ridan chaqiriladi)
# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[float], q: float) -> float:
    """Lineer-interpolatsiyali persentil (numpy default'iga mos). q: 0..100.

    `sorted_vals` o'sish tartibida bo'lishi SHART. Bo'sh -> 0.0.
    """
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def summarize(samples_ms: list[float]) -> dict:
    """Latency namunalaridan statistika. Determinizm: faqat kiruvchi songa bog'liq."""
    if not samples_ms:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(float(x) for x in samples_ms)
    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 3),
        "p50": round(percentile(s, 50), 3),
        "p95": round(percentile(s, 95), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


def check_budget(p95_ms: float, budget_ms: float = DEFAULT_BUDGET_MS) -> tuple[bool, str]:
    """Gate: p95 <= budget bo'lishi SHART. (passed, sabab)."""
    if p95_ms <= budget_ms:
        return True, f"p95 {p95_ms:.1f}ms <= budjet {budget_ms:.0f}ms"
    return False, f"p95 {p95_ms:.1f}ms > budjet {budget_ms:.0f}ms (real-time kritik yo'l buzildi)"


def synth_image(h: int = 720, w: int = 960, seed: int = 0) -> np.ndarray:
    """Deterministik sintetik X-ray-simon RGB rasm (HxWx3 uint8)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def synth_yolo_output(
    num_classes: int = 7, num_preds: int = 8400, imgsz: int = 640, seed: int = 0
) -> np.ndarray:
    """Deterministik sintetik YOLOv11 xom chiqishi: [1, 4+nc, N].

    Real chiqishga o'xshash shaklda (postprocess'ni to'liq ishlatish uchun):
    bir nechta kuchli detection + ko'p past-skorli shovqin -> NMS yuki realistik.
    """
    rng = np.random.RandomState(seed)
    rows = 4 + num_classes
    out = np.zeros((rows, num_preds), dtype=np.float32)
    # boxes (cx, cy, w, h) — imgsz fazosida.
    out[0] = rng.uniform(0, imgsz, num_preds)  # cx
    out[1] = rng.uniform(0, imgsz, num_preds)  # cy
    out[2] = rng.uniform(8, 120, num_preds)    # w
    out[3] = rng.uniform(8, 120, num_preds)    # h
    # class ehtimolliklari — asosan past (shovqin), bir nechtasi baland.
    out[4:] = rng.uniform(0.0, 0.15, size=(num_classes, num_preds)).astype(np.float32)
    strong = rng.choice(num_preds, size=min(12, num_preds), replace=False)
    for i in strong:
        c = rng.randint(0, num_classes)
        out[4 + c, i] = rng.uniform(0.6, 0.95)
    return out[None]  # [1, 4+nc, N]


# ---------------------------------------------------------------------------
# Bosqich timer'lari (backend pre/post-process'ni lazy import qiladi)
# ---------------------------------------------------------------------------
def _import_postprocess():
    """backend/app/pipelines/yolo_postprocess dan funksiyalar. Topilmasa -> None."""
    backend = Path(__file__).resolve().parents[1] / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    try:
        from app.pipelines.yolo_postprocess import letterbox, postprocess, to_input_tensor
    except ImportError:
        return None
    return letterbox, to_input_tensor, postprocess


def _time_loop(fn, iters: int, warmup: int) -> list[float]:
    """fn'ni warmup marta isitib, keyin iters marta o'lchaydi (ms ro'yxati)."""
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def bench_preprocess(image: np.ndarray, imgsz: int, iters: int, warmup: int, pp) -> list[float]:
    letterbox, to_input_tensor, _ = pp

    def step():
        padded, _gain, _pad = letterbox(image, imgsz)
        to_input_tensor(padded)

    return _time_loop(step, iters, warmup)


def bench_postprocess(
    raw: np.ndarray, imgsz: int, orig_w: int, orig_h: int,
    conf: float, iou: float, iters: int, warmup: int, pp,
) -> list[float]:
    _, _, postprocess = pp
    # gain/pad — sintetik raw imgsz fazosida deb hisoblaymiz.
    gain = min(imgsz / orig_h, imgsz / orig_w)

    def step():
        postprocess(raw, gain=gain, pad=(0.0, 0.0), orig_w=orig_w, orig_h=orig_h,
                    conf_thres=conf, iou_thres=iou)

    return _time_loop(step, iters, warmup)


def bench_detect_e2e(model_path: str, image_path: str, iters: int, warmup: int) -> list[float]:
    """Haqiqiy uchidan-uchiga detect() (onnxruntime + model SHART)."""
    backend = Path(__file__).resolve().parents[1] / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from app.pipelines.detection import OnnxYoloDetector

    det = OnnxYoloDetector(model_path)
    det._ensure_session()  # warmup'dan oldin session/provider tanlovini majburla
    return _time_loop(lambda: det.detect(image_path), iters, warmup)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_stage(name: str, stats: dict) -> None:
    if stats["n"] == 0:
        print(f"  {name:<14} o'lchanmadi")
        return
    print(f"  {name:<14} p50 {stats['p50']:>7.2f}ms  p95 {stats['p95']:>7.2f}ms  "
          f"mean {stats['mean']:>7.2f}ms  max {stats['max']:>7.2f}ms  (n={stats['n']})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Detection latency benchmark (qabul mezoni #1: <300ms)")
    ap.add_argument("--model", help="model.onnx yo'li (yo'q bo'lsa numpy-only rejim)")
    ap.add_argument("--image", help="X-ray rasm yo'li (yo'q bo'lsa sintetik)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    args = ap.parse_args()

    pp = _import_postprocess()
    if pp is None:
        print("backend yolo_postprocess topilmadi (sys.path?).", file=sys.stderr)
        return 2

    have_ort = False
    try:
        import onnxruntime  # noqa: F401
        have_ort = True
    except ImportError:
        pass

    full_mode = bool(args.model) and Path(args.model).exists() and have_ort and bool(args.image)

    if full_mode:
        print(f"[TO'LIQ rejim] model={args.model} image={args.image} "
              f"iters={args.iters} warmup={args.warmup}")
        e2e = summarize(bench_detect_e2e(args.model, args.image, args.iters, args.warmup))
        _print_stage("detect() e2e", e2e)
        gate_stat = e2e
    else:
        reason = []
        if not args.model or not Path(args.model or "").exists():
            reason.append("model yo'q")
        if not have_ort:
            reason.append("onnxruntime yo'q")
        if not args.image:
            reason.append("rasm yo'q")
        print(f"[NUMPY-ONLY rejim] ({', '.join(reason)}) — pre/post-process o'lchanadi, "
              "inference o'lchanmaydi.")
        if args.image and Path(args.image).exists():
            from PIL import Image
            with Image.open(args.image) as im:
                image = np.asarray(im.convert("RGB"))
        else:
            image = synth_image(seed=0)
        orig_h, orig_w = image.shape[:2]
        raw = synth_yolo_output(num_classes=7, imgsz=args.imgsz, seed=0)

        pre = summarize(bench_preprocess(image, args.imgsz, args.iters, args.warmup, pp))
        post = summarize(bench_postprocess(raw, args.imgsz, orig_w, orig_h, args.conf,
                                           args.iou, args.iters, args.warmup, pp))
        _print_stage("preprocess", pre)
        _print_stage("inference", {"n": 0})
        _print_stage("postprocess", post)
        # numpy-only'da budjet faqat o'lchangan bosqichlar yig'indisiga nisbatan
        # (inference HALI qo'shilmagan) — gate "qisman" deb belgilanadi.
        combined_p95 = pre["p95"] + post["p95"]
        gate_stat = {"p95": combined_p95}
        print(f"  {'pre+post p95':<14} {combined_p95:>7.2f}ms  "
              "(inference budjeti ALOHIDA — model kelganda to'liq o'lchanadi)")

    passed, reason = check_budget(gate_stat["p95"], args.budget_ms)
    tag = "PASS" if passed else "FAIL"
    print(f"\n[GATE {tag}] {reason}")
    if not full_mode:
        print("  ⚠️ Bu QISMAN gate (inference o'lchanmadi). To'liq tasdiq uchun real "
              "model + onnxruntime bilan --model --image bering.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
