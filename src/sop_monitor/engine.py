"""Model loading, drawing, and the background video-verification worker.

A category counts as DETECTED once it appears in at least `min_frames`
frames with confidence >= `min_conf` (see config.load_sop_rules). Assembly
rules add a spatial check on top: an "inner" component must overlap its
"outer" component by at least `min_overlap` of the inner box's area in
`min_frames` frames (e.g. the main welding plate sitting in the copper).
Works with oriented-bounding-box (OBB) models and regular axis-aligned
models alike: OBB results get the custom polygon drawing below, anything
else falls back to Ultralytics' own `result.plot()`.
"""

from __future__ import annotations

import base64
import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import settings

PREVIEW_EVERY = 15        # frames between live-preview refreshes
PREVIEW_MAX_W = 480       # px; keeps the base64 payload small enough to poll

CLASS_COLORS = [
    (110, 231, 183),
    (147, 197, 253),
    (252, 165, 165),
    (252, 211, 77),
    (196, 181, 253),
]

_model: YOLO | None = None
_model_path: Path | None = None
model_lock = threading.Lock()

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def get_model() -> tuple[YOLO, Path]:
    """Load the checkpoint once and reuse it. Predictions are serialized
    through model_lock because YOLO.predict isn't thread-safe on a shared
    model instance."""
    global _model, _model_path
    with model_lock:
        if _model is None:
            _model_path = settings.resolve_weights()
            _model = YOLO(str(_model_path))
        return _model, _model_path


def model_names() -> dict[int, str]:
    model, _ = get_model()
    return {int(k): v for k, v in model.names.items()}


def device_str() -> str:
    return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"


def new_class_states(names: dict[int, str], required_ids: list[int]) -> list[dict]:
    return [{
        "id": cid,
        "name": names[cid],
        "required": cid in required_ids,
        "detected": False,
        "frames_seen": 0,
        "best_conf": 0.0,
        "first_seen_frame": None,
        "first_seen_sec": None,
    } for cid in sorted(names)]


def new_rule_states(rules: list[dict]) -> list[dict]:
    return [{
        "inner": r["inner_name"],
        "outer": r["outer_name"],
        "inner_id": r["inner_id"],
        "outer_id": r["outer_id"],
        "min_overlap": r["min_overlap"],
        "satisfied": False,
        "frames_ok": 0,
        "best_overlap": 0.0,
        "first_ok_frame": None,
        "first_ok_sec": None,
    } for r in rules]


def _result_hits(result) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(class_ids, confidences, corner polygons) from an OBB or axis-aligned
    result, else None. Polygons are float32 (N, 4, 2) so the assembly-overlap
    checks work identically for rotated and upright boxes."""
    det = getattr(result, "obb", None)
    is_obb = det is not None
    if det is None:
        det = getattr(result, "boxes", None)
    if det is None or len(det) == 0:
        return None
    cls = det.cls.cpu().numpy().astype(int)
    conf = det.conf.cpu().numpy()
    if is_obb:
        polys = det.xyxyxyxy.cpu().numpy().astype(np.float32)
    else:
        xyxy = det.xyxy.cpu().numpy()
        polys = np.stack([xyxy[:, [0, 1]], xyxy[:, [2, 1]],
                          xyxy[:, [2, 3]], xyxy[:, [0, 3]]],
                         axis=1).astype(np.float32)
    return cls, conf, polys


def _containment(inner_poly: np.ndarray, outer_poly: np.ndarray) -> float:
    """Fraction (0..1) of `inner_poly`'s area covered by `outer_poly`.

    convexHull guarantees the vertex ordering intersectConvexConvex expects,
    regardless of how the model emits the corners."""
    inner = cv2.convexHull(inner_poly.astype(np.float32))
    outer = cv2.convexHull(outer_poly.astype(np.float32))
    inner_area = cv2.contourArea(inner)
    if inner_area <= 0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(inner, outer)
    return float(inter_area) / float(inner_area)


def _update_rules(rule_states: list[dict], cls: np.ndarray, polys: np.ndarray,
                  frame_idx: int, fps: float, min_frames: int) -> None:
    """Per-frame assembly check: a frame counts for a rule when any inner
    detection overlaps any outer detection by at least the rule's min_overlap."""
    for rs in rule_states:
        inner_idx = np.flatnonzero(cls == rs["inner_id"])
        outer_idx = np.flatnonzero(cls == rs["outer_id"])
        if len(inner_idx) == 0 or len(outer_idx) == 0:
            continue
        best = 0.0
        for i in inner_idx:
            for j in outer_idx:
                best = max(best, _containment(polys[i], polys[j]))
        rs["best_overlap"] = round(max(rs["best_overlap"], best), 3)
        if best >= rs["min_overlap"]:
            rs["frames_ok"] += 1
            if rs["first_ok_frame"] is None:
                rs["first_ok_frame"] = frame_idx
                rs["first_ok_sec"] = round(frame_idx / fps, 2)
            if not rs["satisfied"] and rs["frames_ok"] >= min_frames:
                rs["satisfied"] = True


def _draw_obb(frame: np.ndarray, result) -> np.ndarray:
    """Draw OBB polygons + labels on `frame` in-place and return it.

    Line thickness, font scale, and label padding all scale with image size
    so boxes stay legible on 4k photos and small frames alike."""
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return frame

    h, w = frame.shape[:2]
    short_side = min(h, w)
    box_thick   = max(3, short_side // 250)
    outline     = box_thick + 2
    font_scale  = max(0.6, short_side / 900.0)
    text_thick  = max(2, int(round(font_scale * 2)))
    pad_x       = max(6, int(round(font_scale * 8)))
    pad_y       = max(4, int(round(font_scale * 6)))

    polys = obb.xyxyxyxy.cpu().numpy().astype(np.int32)
    cls = obb.cls.cpu().numpy().astype(int)
    conf = obb.conf.cpu().numpy()
    names = result.names

    for i in range(len(polys)):
        color = CLASS_COLORS[int(cls[i]) % len(CLASS_COLORS)]
        pts = polys[i].reshape(-1, 2)
        cv2.polylines(frame, [pts], True, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, color, box_thick, cv2.LINE_AA)

        label = f"{names.get(int(cls[i]), int(cls[i]))} {conf[i]:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)

        anchor_idx = int(np.argmin(pts[:, 1]))
        ax, ay = pts[anchor_idx]
        box_x = int(ax)
        box_y = int(ay) - th - 2 * pad_y
        if box_y < 0:
            box_y = int(ay) + 2
        box_x = max(0, min(box_x, w - tw - 2 * pad_x - 1))

        x0, y0 = box_x, box_y
        x1, y1 = box_x + tw + 2 * pad_x, box_y + th + 2 * pad_y
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, label,
                    (x0 + pad_x, y1 - pad_y - baseline // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), text_thick, cv2.LINE_AA)
    return frame


def _annotate(frame: np.ndarray, result) -> np.ndarray:
    if getattr(result, "obb", None) is not None:
        return _draw_obb(frame, result)
    return result.plot()


def _open_video_writer(out_path: Path, fps: float, w: int, h: int):
    """Try a browser-friendly H.264 codec first, fall back to MPEG-4 Part 2.

    Windows builds of OpenCV can usually encode avc1 directly; the Linux/macOS
    pip wheels ship without an H.264 encoder, so they land on mp4v here and
    the finished file gets transcoded by _transcode_h264 instead."""
    with _quiet_stderr():
        for fourcc_code in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            if writer.isOpened():
                return writer, fourcc_code
            writer.release()
    raise RuntimeError("no usable VideoWriter codec found")


@contextlib.contextmanager
def _quiet_stderr():
    """Silence OS-level stderr for the duration of the block.

    OpenCV's codec probing prints harmless-but-scary errors from native code
    (the OpenH264 loader writes to the C stderr directly, and the videoio
    plugin DLL ignores cv2's Python log level), so plain logging knobs can't
    suppress them — only redirecting file descriptor 2 does."""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stderr.flush()
    saved = os.dup(fd)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


def _ffmpeg_exe() -> str | None:
    """System ffmpeg if on PATH, else the binary bundled by imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_h264(path: Path) -> bool:
    """Re-encode `path` in place to browser-playable H.264. Returns success."""
    exe = _ffmpeg_exe()
    if not exe:
        return False
    tmp = path.with_name(path.stem + ".h264.mp4")
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(path),
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        tmp.replace(path)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def _frame_preview(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    if w > PREVIEW_MAX_W:
        frame = cv2.resize(frame, (PREVIEW_MAX_W, int(h * PREVIEW_MAX_W / w)))
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii")


def process_video_job(job_id: str, src_path: Path) -> None:
    job = jobs[job_id]
    sop = job["sop"]
    try:
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError("could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        job["fps"] = round(fps, 2)
        job["total_frames"] = total

        out_path = settings.processed_dir / f"{job_id}.mp4"
        writer, codec = _open_video_writer(out_path, fps, w, h)
        job["codec"] = codec

        model, _ = get_model()
        device = 0 if torch.cuda.is_available() else "cpu"
        by_id = {c["id"]: c for c in job["classes"]}
        rule_states = job.get("rules") or []
        min_frames = sop["min_frames"]
        frame_idx = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with model_lock:
                r = model.predict(frame, imgsz=640, conf=sop["min_conf"], iou=0.45,
                                  device=device, verbose=False)[0]

            hits = _result_hits(r)
            if hits is not None:
                cls, conf, polys = hits
                for cid in set(cls.tolist()):
                    st = by_id.get(cid)
                    if st is None:
                        continue
                    st["frames_seen"] += 1
                    st["best_conf"] = round(max(st["best_conf"],
                                                float(conf[cls == cid].max())), 3)
                    if st["first_seen_frame"] is None:
                        st["first_seen_frame"] = frame_idx
                        st["first_seen_sec"] = round(frame_idx / fps, 2)
                    if not st["detected"] and st["frames_seen"] >= min_frames:
                        st["detected"] = True
                if rule_states:
                    _update_rules(rule_states, cls, polys, frame_idx, fps,
                                  min_frames)

            frame = _annotate(frame, r)
            writer.write(frame)
            frame_idx += 1
            if total > 0 and frame_idx % 5 == 0:
                job["progress"] = round(frame_idx / total, 4)
                job["frames"] = frame_idx
            if frame_idx % PREVIEW_EVERY == 1:
                job["preview"] = _frame_preview(frame)
        cap.release()
        writer.release()

        if codec == "mp4v":
            job["progress"] = 0.999  # keep the bar honest during transcode
            if _transcode_h264(out_path):
                job["codec"] = "h264 (ffmpeg)"
            else:
                job["codec_warning"] = (
                    "Result was encoded as mp4v (no H.264 encoder found); it may "
                    "not play inline in the browser — use the download link, or "
                    "install ffmpeg / `pip install imageio-ffmpeg` and re-run."
                )

        missing = [c["name"] for c in job["classes"]
                   if c["required"] and not c["detected"]]
        failed_rules = [f'{r["inner"]} in {r["outer"]}'
                        for r in rule_states if not r["satisfied"]]
        job["progress"] = 1.0
        job["frames"] = frame_idx
        job["elapsed_sec"] = round(time.time() - t0, 2)
        job["missing"] = missing
        job["failed_rules"] = failed_rules
        job["verdict"] = "PASS" if not missing and not failed_rules else "FAIL"
        job["output_url"] = f"/processed/{out_path.name}"
        job["status"] = "done"
        try:
            src_path.unlink()
        except OSError:
            pass
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
