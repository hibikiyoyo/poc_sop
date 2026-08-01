# AI SOP — Component Verification by Video

Standard Operating Procedure for verifying, with a YOLO detector, that every
required object category appears in an inspection video.

| | |
|---|---|
| **System**   | AI SOP Monitor (`sop-monitor`, http://127.0.0.1:8000/) |
| **Model**    | Ultralytics YOLO checkpoint at `weights/best.pt` |
| **Scope**    | Recorded inspection videos |
| **Owner**    | BOYD |
| **Doc**      | SOP.md v1.0 — 2026-08-01 |

## 1. Purpose

Guarantee that an inspection video shows **all required categories**. The
monitor runs the detector over every frame, tracks which categories have
been seen, and raises an alert at the end of the video for any category that
never appeared.

## 2. Target categories

Categories come from the model. The bundled welding-plate checkpoint knows
five classes, all required by default:

| id | class               | required |
|----|---------------------|----------|
| 0  | copper              | ✔ |
| 1  | main welding plate  | ✔ |
| 2  | cover               | ✔ |
| 3  | cover welding plate | ✔ |
| 4  | final               | ✔ |

To require only a subset, list it in `sop_config.json` (§6).

## 3. Detection & verification rules

- A detection counts only when its confidence is **≥ `min_conf`** (default **0.25**).
- A category is marked **DETECTED** once it appears in **≥ `min_frames`**
  frames (default **3**) — this filters out single-frame false positives.
- **PASS**: every required category is DETECTED when the video ends.
- **FAIL**: one or more required categories were never detected → the
  dashboard shows a red **SOP ALERT** banner listing the missing objects and
  plays an audible alarm.

## 4. Operating procedure

1. **Start the monitor**
   ```
   make run            # or: .venv/Scripts/sop-monitor.exe  (Windows)
                       #     .venv/bin/sop-monitor          (Linux/macOS)
   ```
   Open http://127.0.0.1:8000/ — the header must show the expected weights
   file and `cuda:0` (GPU) as the device.
2. **Check the SOP rules panel** — it lists the required categories and the
   active thresholds. If they don't match this SOP, fix `sop_config.json`
   (§6) before continuing.
3. **Upload the inspection video** (drag & drop or click). Processing starts
   immediately; the live preview shows the annotated frames as they are
   analyzed.
4. **Watch the category checklist.** Each card flips from `… PENDING` to
   `✔ DETECTED` the moment its rule in §3 is satisfied, and records first-seen
   time, frame count, and best confidence.
5. **Read the verdict** when progress reaches 100 %:
   - ✅ **SOP PASSED** — archive the annotated video (link on the dashboard)
     as inspection evidence.
   - 🚨 **SOP ALERT** — one or more categories missing; follow §5.

## 5. Alert handling

When the monitor reports missing categories:

1. Open the annotated result video and confirm the object truly never
   appears (rule out camera framing that cut it off).
2. If the component is genuinely absent → **stop the line / flag the
   assembly** and notify the process owner.
3. If the component is visible to the eye but not detected → re-record with
   better lighting/angle and re-run. If it still fails, treat it as a model
   gap: collect and label the failing frames, then retrain.
4. Record the outcome (video file, verdict, action taken) in the inspection
   log.

## 6. Configuration — `sop_config.json` (optional)

Copy `sop_config.example.json` to `sop_config.json` and edit; the monitor
re-reads it for every uploaded video (no restart needed):

```json
{
  "required_classes": ["copper", "main welding plate", "cover",
                        "cover welding plate", "final"],
  "min_conf": 0.25,
  "min_frames": 3
}
```

- `required_classes` — class names or numeric ids; omit (or leave invalid)
  to require **all** model classes. Categories not listed still appear on
  the dashboard as `OPTIONAL` and never trigger alerts.
- `min_conf` — confidence gate for a detection to count.
- `min_frames` — frames a category must be seen in before it is DETECTED.

## 7. Maintenance

- **Weights**: replace `weights/best.pt` (or set `SOP_MONITOR_WEIGHTS`) and
  restart the monitor.
- **Retraining trigger**: recurring false alerts on a category (§5.3).
- **Housekeeping**: `make clean` empties `monitor_data/` (uploads +
  annotated results). Archive evidence videos first.
