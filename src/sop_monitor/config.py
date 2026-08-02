"""Runtime settings (environment variables) and SOP rule loading.

Everything is overridable via environment variables so the app can run from
any working directory:

    SOP_MONITOR_HOST      bind address           (default 127.0.0.1)
    SOP_MONITOR_PORT      port                   (default 8000)
    SOP_MONITOR_WEIGHTS   path to a .pt file     (default: weights/best.pt,
                                                  else newest weights/*.pt)
    SOP_MONITOR_DATA      uploads/results dir    (default ./monitor_data)
    SOP_MONITOR_CONFIG    SOP rules json         (default ./sop_config.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        root = Path(os.environ.get("SOP_MONITOR_HOME", os.getcwd()))
        self.root = root
        self.host = os.environ.get("SOP_MONITOR_HOST", "127.0.0.1")
        self.port = int(os.environ.get("SOP_MONITOR_PORT", "8000"))
        self.weights_env = os.environ.get("SOP_MONITOR_WEIGHTS", "")
        self.weights_dir = root / "weights"
        self.data_dir = Path(os.environ.get("SOP_MONITOR_DATA", root / "monitor_data"))
        self.uploads_dir = self.data_dir / "uploads"
        self.processed_dir = self.data_dir / "processed"
        self.sop_config_path = Path(os.environ.get("SOP_MONITOR_CONFIG",
                                                   root / "sop_config.json"))

    def resolve_weights(self) -> Path:
        """Pick the model checkpoint: env var > weights/best.pt > newest weights/*.pt."""
        if self.weights_env:
            p = Path(self.weights_env)
            if p.exists():
                return p
            raise SystemExit(f"SOP_MONITOR_WEIGHTS points to a missing file: {p}")
        default = self.weights_dir / "best.pt"
        if default.exists():
            return default
        candidates = sorted(self.weights_dir.glob("*.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
        raise SystemExit(
            "No model weights found. Put your trained checkpoint at "
            "weights/best.pt or set SOP_MONITOR_WEIGHTS=path/to/model.pt"
        )


settings = Settings()


def _parse_assembly_rules(raw, names: dict[int, str],
                          by_name: dict[str, int]) -> list[dict]:
    """Resolve assembly_rules entries ({"inner", "outer", "min_overlap"?}) to
    class ids. Rules naming unknown classes are warned about and skipped."""
    rules: list[dict] = []
    if raw is None:
        return rules
    if not isinstance(raw, list):
        print("Warning: sop_config.json assembly_rules must be a list — ignored")
        return rules
    for item in raw:
        if not isinstance(item, dict):
            print(f"Warning: skipping invalid assembly rule {item!r}")
            continue
        ids = []
        for key in ("inner", "outer"):
            v = item.get(key)
            if isinstance(v, int) and v in names:
                ids.append(v)
            elif isinstance(v, str) and v in by_name:
                ids.append(by_name[v])
            else:
                print(f"Warning: assembly rule {key!r} names unknown class {v!r}")
        if len(ids) != 2 or ids[0] == ids[1]:
            continue
        try:
            min_overlap = float(item.get("min_overlap", 0.5))
        except (TypeError, ValueError):
            min_overlap = 0.5
        rules.append({
            "inner_id": ids[0], "inner_name": names[ids[0]],
            "outer_id": ids[1], "outer_name": names[ids[1]],
            "min_overlap": min(max(min_overlap, 0.0), 1.0),
        })
    return rules


def load_sop_rules(names: dict[int, str]) -> dict:
    """Merge sop_config.json (if present) over the defaults and resolve the
    required classes to ids. Unknown entries are warned about and skipped;
    an empty/invalid list falls back to requiring every model class."""
    cfg = {"required_classes": None, "min_conf": 0.25, "min_frames": 3,
           "assembly_rules": None}
    if settings.sop_config_path.exists():
        try:
            user = json.loads(settings.sop_config_path.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in user.items() if k in cfg})
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: ignoring invalid {settings.sop_config_path.name}: {e}")

    by_name = {v: k for k, v in names.items()}
    required = cfg["required_classes"]
    if required is None:
        ids = sorted(names)
    else:
        ids = []
        for item in required:
            if isinstance(item, int) and item in names:
                ids.append(item)
            elif isinstance(item, str) and item in by_name:
                ids.append(by_name[item])
            else:
                print(f"Warning: sop_config.json names unknown class {item!r}")
        ids = sorted(set(ids)) or sorted(names)
    return {
        "min_conf": float(cfg["min_conf"]),
        "min_frames": max(1, int(cfg["min_frames"])),
        "required_ids": ids,
        "required_names": [names[i] for i in ids],
        "assembly_rules": _parse_assembly_rules(cfg["assembly_rules"],
                                                names, by_name),
    }
