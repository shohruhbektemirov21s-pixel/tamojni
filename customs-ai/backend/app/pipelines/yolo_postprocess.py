"""YOLOv11 pre/post-processing — sof numpy (torch'siz, opencv'siz).

ADR-002: runtime'da torch YO'Q. Bu modul faqat numpy'ga tayanadi, shuning uchun
onnxruntime/model bo'lmasa ham mustaqil test qilinadi (deterministik birliklar).

YOLOv11 ONNX chiqishi: shakl [1, 4 + nc, N] (transpose qilinmagan), bu yerda
birinchi 4 qator = (cx, cy, w, h) model kirish koordinatasida, qolgan nc qator =
har class uchun sigmoid'lanmagan EMAS — Ultralytics eksportida allaqachon
ehtimollik (0..1). Biz sigmoid qo'llamaymiz (Ultralytics default eksport shartiga
mos). Agar boshqa eksport ishlatilsa, `assume_probabilities=False` bilan sigmoid
yoqiladi.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "letterbox",
    "decode_yolo_output",
    "nms_per_class",
    "scale_boxes",
    "postprocess",
]


def letterbox(
    img: np.ndarray, new_shape: int = 640, color: int = 114
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Nisbatni saqlab o'lchamga keltirish + padding.

    img: HxWx3 uint8 (RGB). Qaytaradi: (padded HxWx3, gain, (pad_x, pad_y)).
    gain va pad keyin bbox'ni asl rasm koordinatasiga qaytarish uchun kerak.
    """
    h, w = img.shape[:2]
    gain = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * gain)), int(round(w * gain))

    # Nearest-neighbour resize (numpy, opencv'siz) — determinizm uchun yetarli.
    yi = (np.arange(nh) / gain).astype(np.int64).clip(0, h - 1)
    xi = (np.arange(nw) / gain).astype(np.int64).clip(0, w - 1)
    resized = img[yi][:, xi]

    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    pad_x = (new_shape - nw) / 2
    pad_y = (new_shape - nh) / 2
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, gain, (float(left), float(top))


def to_input_tensor(padded: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 (RGB) -> [1,3,H,W] float32 [0,1] (NCHW)."""
    x = padded.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # HWC -> CHW -> NCHW
    return np.ascontiguousarray(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolo_output(
    output: np.ndarray, conf_thres: float, assume_probabilities: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLOv11 xom chiqishini (boxes_xyxy, scores, class_ids) ga aylantiradi.

    output: [1, 4+nc, N] yoki [4+nc, N]. Koordinata model kirish fazosida (xyxy).
    conf_thres pollikdan past bo'lganlar tashlanadi.
    """
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    # [4+nc, N] -> [N, 4+nc]
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] < 5:
        return _empty()

    boxes = arr[:, :4]
    cls_scores = arr[:, 4:]
    if not assume_probabilities:
        cls_scores = _sigmoid(cls_scores)

    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    keep = scores >= conf_thres
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    if boxes.shape[0] == 0:
        return _empty()

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return xyxy, scores, class_ids


def _empty() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), np.float32),
        np.zeros((0,), np.float32),
        np.zeros((0,), np.int64),
    )


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def nms_per_class(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thres: float
) -> list[int]:
    """Class bo'yicha NMS. Saqlanadigan indekslar (asl massivga nisbatan).

    Deterministik: teng score'larda indeks bo'yicha barqaror tartib.
    """
    keep: list[int] = []
    for c in np.unique(class_ids):
        idxs = np.where(class_ids == c)[0]
        # score kamayish + indeks o'sish bo'yicha barqaror tartib (determinizm).
        order = sorted(idxs.tolist(), key=lambda i: (-float(scores[i]), int(i)))
        while order:
            i = order.pop(0)
            keep.append(i)
            if not order:
                break
            rest = np.array(order)
            ious = _iou(boxes[i], boxes[rest])
            order = [int(j) for j, iou in zip(order, ious) if iou <= iou_thres]
    keep.sort()
    return keep


def scale_boxes(
    boxes: np.ndarray, gain: float, pad: tuple[float, float], orig_w: int, orig_h: int
) -> np.ndarray:
    """Model kirish koordinatasidagi xyxy'ni asl rasm koordinatasiga qaytaradi."""
    if boxes.shape[0] == 0:
        return boxes
    pad_x, pad_y = pad
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / gain
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / gain
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, orig_h)
    return out


def postprocess(
    output: np.ndarray,
    gain: float,
    pad: tuple[float, float],
    orig_w: int,
    orig_h: int,
    conf_thres: float,
    iou_thres: float,
    assume_probabilities: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """To'liq post-process: decode -> NMS -> asl koordinataga scale.

    Qaytaradi: (boxes_xyxy[int-able float], scores, class_ids) asl rasm fazosida.
    """
    boxes, scores, class_ids = decode_yolo_output(output, conf_thres, assume_probabilities)
    if boxes.shape[0] == 0:
        return boxes, scores, class_ids
    keep = nms_per_class(boxes, scores, class_ids, iou_thres)
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    boxes = scale_boxes(boxes, gain, pad, orig_w, orig_h)
    return boxes, scores, class_ids
