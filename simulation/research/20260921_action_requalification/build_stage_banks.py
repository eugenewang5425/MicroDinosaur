"""Split the audited diverse reset bank into side acquisition and mixed retention banks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "side_getup_reset_bank_v071_r7s.json"
SIDE_DEST = HERE / "side_getup_reset_bank_v071_r7s_stage1.json"
MIXED_DEST = HERE / "side_getup_reset_bank_v071_r7s_stage2.json"


def write_bank(destination: Path, source: dict, records: list[dict], stage: str) -> None:
    payload = dict(source)
    payload["records"] = records
    payload["sample_count"] = len(records)
    payload["per_label_counts"] = dict(Counter(row["label"] for row in records))
    payload["training_stage"] = stage
    payload["source_bank"] = SOURCE.name
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = payload["records"]
    side = [row for row in records if row["label"].split("_")[0] in {"left", "right"}]
    if not side or not any(row.get("state_class") == "full_fall" for row in side):
        raise ValueError("Side bank has no audited full-fall samples")
    write_bank(SIDE_DEST, payload, side, "side_acquisition")
    write_bank(MIXED_DEST, payload, records, "mixed_retention")
    print(json.dumps({"source": len(records), "stage1": len(side), "stage2": len(records)}))


if __name__ == "__main__":
    main()
