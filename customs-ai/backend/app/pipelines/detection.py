"""Vision detection — EGASI: Dev 2.

Backend Core faqat `Detector` kontraktini chaqiradi (contracts.py).
ADR-002: GPU'da PyTorch ishlatilMAYDI — ONNX Runtime (CPU yoki DirectML).
GPU (4GB) faqat Ollama'niki; ikki VRAM menejeri urishmasligi uchun YOLO
CPU/DirectML'da ishlaydi. X-ray'da kichik YOLOv11 + CPU = ~100-300ms.

⚠️ DOMAIN GAP (xavf #1): bu detector FAQAT X-ray'da fine-tune qilingan model bilan
to'g'ri ishlaydi. COCO/tabiiy rasm weights X-ray'da ishonchsiz. Model
`model_registry/<versiya>/model.onnx` + yonidagi `labels.txt` dan yuklanadi.

Provider tartibi (ADR-002):
    DmlExecutionProvider (DirectML) -> CPUExecutionProvider
CUDAExecutionProvider ATAYIN ishlatilmaydi (GPU Ollama'ga atalgan).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.pipelines.yolo_postprocess import letterbox, postprocess, to_input_tensor

log = logging.getLogger("customs.detection")

# ADR-002: CUDA YO'Q. DirectML bo'lsa undan, bo'lmasa CPU'dan foydalanamiz.
_PREFERRED_PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]


def _select_providers(available: list[str]) -> list[str]:
    chosen = [p for p in _PREFERRED_PROVIDERS if p in available]
    return chosen or ["CPUExecutionProvider"]


def _load_labels(labels_path: Path) -> list[str]:
    with labels_path.open("r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


class OnnxYoloDetector:
    """YOLOv11 -> ONNX Runtime inference (X-ray fine-tuned weights).

    Parametrlar:
        model_path: .onnx fayl yo'li.
        labels: class nomlari ro'yxati YOKI labels_path orqali fayldan.
        conf: detektsiya pollik (PAST qo'yiladi — recall foydasiga; yakuniy
              per-class pollik risk engine'da). iou: NMS pollik.
        imgsz: model kirish o'lchami (kvadrat).
    """

    def __init__(
        self,
        model_path: str,
        *,
        labels: list[str] | None = None,
        labels_path: str | None = None,
        conf: float = 0.20,
        iou: float = 0.50,
        imgsz: int = 640,
        assume_probabilities: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.assume_probabilities = assume_probabilities

        if labels is not None:
            self.labels = list(labels)
        elif labels_path is not None:
            self.labels = _load_labels(Path(labels_path))
        else:
            sibling = Path(self.model_path).with_name("labels.txt")
            if sibling.exists():
                self.labels = _load_labels(sibling)
            else:
                raise FileNotFoundError(
                    f"labels topilmadi: {sibling} (yoki labels=/labels_path= bering). "
                    "Class nomlarisiz detection class'larni taksonomiyaga moslay olmaydi."
                )

        self._session = None  # lazy: onnxruntime import faqat kerak bo'lganda
        self._input_name: str | None = None

    # ---- session (lazy) ----
    def _ensure_session(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort  # lazy: backend core onnxruntime'siz ishlaydi
        except ImportError as exc:  # pragma: no cover - muhitga bog'liq
            raise RuntimeError(
                "onnxruntime o'rnatilmagan. offline-wheels orqali o'rnating "
                "(onnxruntime-directml Windows'da, onnxruntime Linux'da)."
            ) from exc

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX model topilmadi: {self.model_path}")

        available = ort.get_available_providers()
        providers = _select_providers(available)
        log.info("ONNX YOLO yuklanmoqda: %s | providers=%s", self.model_path, providers)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0  # ORT o'zi tanlaydi (Ryzen 5 7000 yadrolari)
        self._session = ort.InferenceSession(self.model_path, sess_options=so, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        return self._session

    @property
    def active_providers(self) -> list[str]:
        return list(self._session.get_providers()) if self._session else []

    # ---- inference ----
    def detect(self, image_path: str) -> list[dict]:
        """-> [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]

        bbox = asl rasm piksellaridagi butun koordinatalar. confidence 0..1.
        Determinizm: bir xil rasm + model -> bir xil chiqish (NMS barqaror tartibli).
        """
        session = self._ensure_session()

        with Image.open(image_path) as im:
            # X-ray psevdo-rang yoki gray bo'lishi mumkin — 3 kanalga keltiramiz.
            rgb = np.asarray(im.convert("RGB"))
        orig_h, orig_w = rgb.shape[:2]

        padded, gain, pad = letterbox(rgb, self.imgsz)
        tensor = to_input_tensor(padded)
        outputs = session.run(None, {self._input_name: tensor})

        boxes, scores, class_ids = postprocess(
            outputs[0],
            gain=gain,
            pad=pad,
            orig_w=orig_w,
            orig_h=orig_h,
            conf_thres=self.conf,
            iou_thres=self.iou,
            assume_probabilities=self.assume_probabilities,
        )

        results: list[dict] = []
        for box, score, cid in zip(boxes, scores, class_ids):
            cid = int(cid)
            cls = self.labels[cid] if 0 <= cid < len(self.labels) else f"class_{cid}"
            x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
            results.append(
                {"class": cls, "confidence": round(float(score), 4), "bbox": [x1, y1, x2, y2]}
            )
        # Auditda barqaror tartib: confidence kamayishi bo'yicha.
        results.sort(key=lambda r: -r["confidence"])
        log.info("detect(%s): %d obyekt", image_path, len(results))
        return results
