# AI SOP Monitor

FastAPI dashboard that verifies inspection videos against an SOP: a YOLO26s-OBB
model (five welding-plate components) checks every frame; any required category
that never appears — or assembly rule (inner-in-outer overlap) never satisfied —
triggers a FAIL verdict and an alert. Internal tool, Python
3.10–3.13 (3.12 pinned via `.python-version`). Windows, Linux, macOS, and
Docker are all supported targets — keep it that way.

## Repo map
- `src/sop_monitor/app.py` — FastAPI routes + `main()` CLI (`--host/--port/--weights/--check`)
- `src/sop_monitor/engine.py` — model loading, OBB drawing, background video worker, codec fallback
- `src/sop_monitor/config.py` — env settings (`SOP_MONITOR_*`) + `sop_config.json` rule loading
- `src/sop_monitor/templates/monitor.html` — the whole frontend (Jinja2 + vanilla JS, no build step)
- `weights/best.pt` — trained checkpoint, intentionally committed (21 MB)
- `Makefile` / `Dockerfile` — cross-platform commands; recipes must stay cmd.exe-safe

## Commands
- Install (GPU): `make install` · CPU/macOS: `make install-cpu`
  (uv users: `uv pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu126`)
- Run: `make run` (`PORT=9000` to override) → http://127.0.0.1:8000/
- Dev with reload: `make dev`
- Sanity check (loads model, prints device + SOP rules, exits): `make gpu-check`
- There is no test suite. Verify with `make gpu-check`, plus one real end-to-end
  video upload for any `engine.py`/`app.py` change.

## Absolute rules
- MUST NOT delete or gitignore `weights/best.pt` — the app must work straight after clone.
- MUST NOT move torch into `[project] dependencies` or "simplify" the extras —
  pip cannot pick CUDA wheels from PyPI; the `[gpu]`/`[cpu]` extras + custom index are deliberate.
- MUST NOT break Makefile shell-agnosticism: no `VAR=x cmd` env prefixes, no
  `rm`/`mkdir`/`cp` in recipes, bare interpreter names in `||` chains — recipes
  run under cmd.exe AND sh.
- MUST NOT remove the mp4v → ffmpeg-transcode fallback or the stderr silencing
  in `engine._open_video_writer` / `_quiet_stderr` (see conventions below).
- MUST NOT commit `sop_config.json`, `monitor_data/`, or `.venv/`.
- MUST keep changes runnable on Linux and Windows: `pathlib` only, no
  OS-specific paths, LF endings per `.gitattributes`.
- MUST run `make gpu-check` and confirm it passes before declaring done.

## Non-obvious conventions
- Plain `uv sync` does NOT install torch (it lives in extras) — use the uv pip
  flow from README. Do not "fix" this by editing dependencies.
- Every `model.predict` call must hold `engine.model_lock` — Ultralytics models
  are not thread-safe on a shared instance.
- Job dicts in `engine.jobs` are mutated by the worker thread while the API
  serializes them — every value must stay JSON-serializable at all times
  (no `datetime`/`Path`/`ndarray` values).
- Codec chain is deliberate: try `avc1`/`H264`, fall back to `mp4v`, then
  transcode with ffmpeg (bundled via `imageio-ffmpeg`) — Linux/macOS OpenCV
  wheels cannot encode H.264. "Failed to load OpenH264" messages during codec
  probing are expected native noise (hence the fd-level stderr silencing), not a bug.
- `monitor.html` is rendered by Jinja2: `{{ }}` is template syntax; the embedded
  JS uses `${}` template literals, which are safe. Keep `templates/*.html`
  listed in `[tool.setuptools.package-data]`.
- SOP rules (`required_classes`, `min_conf`, `min_frames`) are re-read from
  `sop_config.json` on every upload — no restart needed. `sop_config.json` is
  gitignored (local overrides); `sop_config.example.json` is the tracked template.
- `monitor_data/` and `weights/` resolve relative to CWD (overridable via
  `SOP_MONITOR_*` env vars) — run from the repo root.
- The job JSON is the FE/BE contract (`classes[]`, `rules[]`, `verdict`,
  `missing[]`, `failed_rules[]`, `preview`, `output_url`); `monitor.html`
  polls `GET /api/job/{id}` every 0.7 s. Change backend fields and the JS
  renderer together.

## Vocabulary
- **SOP**: the pass/fail procedure in `SOP.md` — every required category must appear in the video.
- **verdict**: `"PASS" | "FAIL"`, computed once at end of video; **missing[]**: required class names never DETECTED.
- **DETECTED**: class seen in ≥ `min_frames` frames at conf ≥ `min_conf`. **PENDING**: not yet. **OPTIONAL**: class not in `required_classes` — never alerts.
- **min_conf** (0.25) / **min_frames** (3): the two accuracy knobs — confidence gate + temporal persistence.
- **assembly rule**: spatial check from `assembly_rules` in `sop_config.json` — ≥ `min_overlap` (default 0.5) of the inner class's box area must overlap an outer-class box in ≥ `min_frames` frames (convex-polygon intersection in `engine._containment`); unsatisfied rules land in **failed_rules[]** and force FAIL.
- **OBB**: oriented bounding box — read from `result.obb`, with `result.boxes` fallback for axis-aligned models.
- **job**: in-memory dict in `engine.jobs`, keyed by 12-hex id; lost on restart (annotated MP4s on disk survive).

## Read when relevant
- `README.md` → architecture, two Mermaid pipeline diagrams, config tables, troubleshooting
- `SOP.md` → the verification rules and alert-handling procedure the app implements
