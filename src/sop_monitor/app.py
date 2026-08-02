"""FastAPI application for the AI SOP monitor.

Run:
    sop-monitor                                (installed console script)
    sop-monitor --port 9000 --host 0.0.0.0     (CLI overrides env settings)
    sop-monitor --check                        (print model/device info, exit)
    python -m sop_monitor.app                  (equivalent)
    uvicorn sop_monitor.app:app --reload       (development, auto-reload)

Endpoints
---------
GET  /                       -> monitor dashboard (Jinja2 template)
GET  /api/meta               -> { names, weights, device, sop: {...} }
POST /api/upload_video       multipart form, field name "file" -> { job_id }
GET  /api/job/{job_id}       -> { status, progress, classes: [...], rules: [...],
                                  preview, verdict?, missing?, failed_rules?,
                                  output_url? }
GET  /processed/<file>       -> annotated MP4 (static mount, Range supported)
GET  /docs                   -> interactive OpenAPI docs (FastAPI built-in)
"""

from __future__ import annotations

import argparse
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__
from .config import load_sop_rules, settings
from .engine import (
    device_str,
    get_model,
    jobs,
    jobs_lock,
    model_names,
    new_class_states,
    new_rule_states,
    process_video_job,
)

settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI SOP Monitor", version=__version__)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/processed", StaticFiles(directory=str(settings.processed_dir)),
          name="processed")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "monitor.html",
                                      {"version": __version__})


@app.get("/api/meta")
def meta():
    _, weights = get_model()
    names = model_names()
    return {
        "weights": str(weights).replace("\\", "/"),
        "device": device_str(),
        "names": names,
        "sop": load_sop_rules(names),
        "version": __version__,
    }


@app.post("/api/upload_video")
def upload_video(file: UploadFile = File(...)):
    names = model_names()
    sop = load_sop_rules(names)

    jid = uuid.uuid4().hex[:12]
    ext = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    src_path = settings.uploads_dir / f"{jid}{ext}"
    with src_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job = {
        "id": jid,
        "status": "running",
        "progress": 0.0,
        "frames": 0,
        "filename": file.filename,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "sop": sop,
        "classes": new_class_states(names, sop["required_ids"]),
        "rules": new_rule_states(sop["assembly_rules"]),
        "preview": None,
    }
    with jobs_lock:
        jobs[jid] = job
    threading.Thread(target=process_video_job, args=(jid, src_path),
                     daemon=True).start()
    return {"job_id": jid}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return job


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI SOP Monitor — YOLO video verification dashboard.")
    parser.add_argument("--host", default=settings.host,
                        help=f"bind address (default {settings.host})")
    parser.add_argument("--port", type=int, default=settings.port,
                        help=f"port (default {settings.port})")
    parser.add_argument("--weights", default="",
                        help="path to a .pt checkpoint (overrides SOP_MONITOR_WEIGHTS)")
    parser.add_argument("--check", action="store_true",
                        help="print model / device / SOP info and exit")
    args = parser.parse_args()
    settings.host, settings.port = args.host, args.port
    if args.weights:
        settings.weights_env = args.weights

    model, weights = get_model()
    names = model_names()
    sop = load_sop_rules(names)
    print(f"Model:  {weights}")
    print(f"Device: {device_str()}")
    print(f"Classes: {model.names}")
    print(f"SOP: require {sop['required_names']} "
          f"(min_conf={sop['min_conf']}, min_frames={sop['min_frames']})")
    for r in sop["assembly_rules"]:
        print(f"SOP assembly rule: {r['inner_name']} in {r['outer_name']} "
              f"(min_overlap={r['min_overlap']})")
    if args.check:
        return
    print(f"SOP monitor listening on http://{settings.host}:{settings.port}/")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
