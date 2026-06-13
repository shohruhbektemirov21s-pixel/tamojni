"""YOLOv11 sof-numpy pre/post-process (onnxruntime/model'siz, deterministik)."""
from __future__ import annotations

import numpy as np

from app.pipelines.yolo_postprocess import (
    decode_yolo_output,
    letterbox,
    nms_per_class,
    postprocess,
    scale_boxes,
    to_input_tensor,
)


def test_letterbox_square_and_aspect():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, gain, (px, py) = letterbox(img, 640)
    assert padded.shape == (640, 640, 3)
    assert gain == 640 / 200  # eng kichik o'lcham bo'yicha
    # 100*gain = 320 balandlik -> vertikal pad (640-320)/2 = 160
    assert abs(py - 160) <= 1
    assert px == 0


def test_to_input_tensor_shape_range():
    padded = np.full((640, 640, 3), 255, dtype=np.uint8)
    t = to_input_tensor(padded)
    assert t.shape == (1, 3, 640, 640)
    assert t.dtype == np.float32
    assert t.max() <= 1.0 and t.min() >= 0.0


def test_decode_filters_by_conf_and_picks_argmax_class():
    # YOLO format [1, 4+nc, N]; N (anchors) >> nc. 8 anchor, 3 class -> [1, 7, 8].
    # anchor0: cx,cy,w,h = 50,50,20,20 ; class scores = [0.1, 0.9, 0.2] -> class 1
    # qolgan anchorlar past skor -> tashlanadi
    out = np.zeros((1, 7, 8), dtype=np.float32)
    out[0, :4, 0] = [50, 50, 20, 20]
    out[0, 4:, 0] = [0.1, 0.9, 0.2]
    out[0, :4, 1] = [10, 10, 5, 5]
    out[0, 4:, 1] = [0.05, 0.1, 0.05]
    boxes, scores, cids = decode_yolo_output(out, conf_thres=0.25)
    assert boxes.shape[0] == 1
    assert cids[0] == 1
    assert abs(scores[0] - 0.9) < 1e-6
    # xyxy: 50-10=40,40,60,60
    np.testing.assert_allclose(boxes[0], [40, 40, 60, 60], atol=1e-4)


def test_decode_handles_feature_major_2d():
    flat = np.zeros((6, 8), dtype=np.float32)  # [4+2, N=8] (batch'siz)
    flat[:4, 0] = [50, 50, 20, 20]
    flat[4:, 0] = [0.9, 0.1]
    boxes, scores, cids = decode_yolo_output(flat, conf_thres=0.25)
    assert boxes.shape[0] == 1 and cids[0] == 0


def test_nms_suppresses_overlap_keeps_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    cids = np.array([0, 0, 0], dtype=np.int64)
    keep = nms_per_class(boxes, scores, cids, iou_thres=0.5)
    assert 0 in keep and 2 in keep and 1 not in keep


def test_nms_per_class_does_not_cross_classes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    cids = np.array([0, 1], dtype=np.int64)  # turli class — ikkalasi qoladi
    keep = nms_per_class(boxes, scores, cids, iou_thres=0.5)
    assert set(keep) == {0, 1}


def test_nms_deterministic_on_ties():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.5, 0.5], dtype=np.float32)
    cids = np.array([0, 0], dtype=np.int64)
    a = nms_per_class(boxes, scores, cids, 0.5)
    b = nms_per_class(boxes, scores, cids, 0.5)
    assert a == b == [0]  # teng score'da kichik indeks yutadi (barqaror)


def test_scale_boxes_inverts_letterbox():
    # 200x100 rasm, letterbox 640 -> gain, pad. Box modelda 40,40,60,60.
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    _, gain, pad = letterbox(img, 640)
    boxes = np.array([[40 + pad[0], 40 + pad[1], 60 + pad[0], 60 + pad[1]]], dtype=np.float32)
    out = scale_boxes(boxes, gain, pad, orig_w=200, orig_h=100)
    np.testing.assert_allclose(out[0], [40 / gain, 40 / gain, 60 / gain, 60 / gain], atol=1e-3)


def test_postprocess_end_to_end():
    out = np.zeros((1, 7, 8), dtype=np.float32)
    out[0, :4, 0] = [320, 320, 40, 40]
    out[0, 4:, 0] = [0.1, 0.95, 0.2]
    boxes, scores, cids = postprocess(
        out, gain=1.0, pad=(0.0, 0.0), orig_w=640, orig_h=640,
        conf_thres=0.25, iou_thres=0.5,
    )
    assert boxes.shape[0] == 1 and cids[0] == 1
    np.testing.assert_allclose(boxes[0], [300, 300, 340, 340], atol=1e-3)


def test_postprocess_empty_below_conf():
    out = np.zeros((1, 7, 8), dtype=np.float32)
    out[0, 4:, 0] = [0.01, 0.02, 0.01]
    boxes, scores, cids = postprocess(
        out, 1.0, (0, 0), 640, 640, conf_thres=0.25, iou_thres=0.5
    )
    assert boxes.shape[0] == 0
