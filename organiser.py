# organiser.py
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scanner import FileRecord

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

DOC_SUBFOLDERS = {
    ".pdf":  "PDF",
    ".docx": "Word",  ".doc":  "Word",
    ".xlsx": "Spreadsheets", ".xls": "Spreadsheets",
    ".pptx": "Presentations", ".ppt": "Presentations",
    ".txt":  "Text", ".csv":  "Text", ".md": "Text",
}


# ── Data model ─────────────────────────────────────────────────

class Move:
    """Represents one planned file operation — src → dst."""
    def __init__(self, src: Path, dst: Path, reason: str):
        self.src    = src
        self.dst    = dst
        self.reason = reason


class Plan:
    """
    A full list of Move objects.
    Nothing touches the filesystem until execute() is called.
    """
    def __init__(self):
        self.moves: List[Move] = []

    def add(self, src: Path, dst: Path, reason: str):
        self.moves.append(Move(src, dst, reason))

    def summary(self) -> str:
        counts = {}
        for move in self.moves:
            counts[move.reason] = counts.get(move.reason, 0) + 1
        lines = [f"  {count:<6} {reason}" for reason, count in sorted(counts.items())]
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    """Same as deduper._md5 — read in chunks, return hex digest."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                block = f.read(65536)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except Exception:
        return ""


def _safe_copy(src: Path, dst: Path) -> bool:
    """
    Copy src → dst with MD5 verification.
    Create parent directories if they don't exist.
    Handle filename collisions by appending _1, _2 etc.
    Return True on success, False on failure.
    """
    try:
        # 1. Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # 2. Handle collisions — find a free filename
        stem    = dst.stem
        suffix  = dst.suffix
        counter = 1
        while dst.exists():
            dst = dst.parent / f"{stem}_{counter}{suffix}"
            counter += 1

        # 3. Copy preserving metadata
        shutil.copy2(src, dst)

        # 4. Verify integrity
        if _md5(src) != _md5(dst):
            dst.unlink()
            return False

        return True

    except Exception:
        return False


def _date_parts(rec: FileRecord) -> Tuple[str, str]:
    """
    Return (year, month_name) from record's capture_date.
    e.g. "2024-01-15" → ("2024", "January")
    Return ("Unknown", "Unknown") if capture_date is None or malformed.
    """
    try:
        parts = rec.capture_date.split("-")
        year  = parts[0]
        month = MONTH_NAMES[int(parts[1])]
        return year, month
    except Exception:
        return "Unknown", "Unknown"


# ── Build plan ─────────────────────────────────────────────────

def build_plan(
    records:       List[FileRecord],
    exact_dupes:   Dict[str, List[FileRecord]],
    near_dupes:    List[List[FileRecord]],      # [best, rest...] per cluster
    face_clusters: Dict,
    output_root:   str,
) -> Plan:
    """
    Decide where every file goes. Build and return a Plan.
    Nothing is copied here — just decisions.
    """
    plan    = Plan()
    out     = Path(output_root)
    assigned: set = set()

    # 1. Exact duplicates — keep first, move rest to Duplicates/Exact/
    for group in exact_dupes.values():
        for rec in group[1:]:
            dst = out / "Duplicates" / "Exact" / rec.path.name
            plan.add(rec.path, dst, "exact duplicate")
            assigned.add(rec.path)

    # 2. Near duplicates — near_dupes is [best, rest...]; move rest only
    for group in near_dupes:
        for rec in group[1:]:
            if rec.path in assigned:
                continue
            dst = out / "Duplicates" / "Similar" / rec.path.name
            plan.add(rec.path, dst, "near duplicate")
            assigned.add(rec.path)

    # 3. Remaining records — organised by type
    for rec in records:
        if rec.path in assigned:
            continue

        if rec.file_type == "screenshot":
            year, _ = _date_parts(rec)
            dst = out / "Screenshots" / year / rec.path.name
            reason = "screenshot"

        elif rec.file_type == "image":
            year, month = _date_parts(rec)
            dst = out / "Photos" / year / month / rec.path.name
            reason = "photo"

        elif rec.file_type == "video":
            year, _ = _date_parts(rec)
            dst = out / "Videos" / year / rec.path.name
            reason = "video"

        elif rec.file_type == "document":
            subfolder = DOC_SUBFOLDERS.get(rec.ext, "Other")
            dst = out / "Documents" / subfolder / rec.path.name
            reason = "document"

        else:
            dst = out / "Other" / rec.path.name
            reason = "other"

        plan.add(rec.path, dst, reason)
        assigned.add(rec.path)

    # # 4. Face clusters — additional copies to People/<name>/
    #    skip noise (key -1)
    for person_name, cluster_records in face_clusters.items():
        if person_name == -1:
            continue
        person_folder = out / "People" / str(person_name)
        for rec in cluster_records:
            dst = person_folder / rec.path.name
            plan.add(rec.path, dst, person_name)

    return plan


# ── Execute ────────────────────────────────────────────────────

def execute(plan: Plan) -> Dict[str, int]:
    """
    Execute every move in the plan.
    Returns {"success": N, "failed": N, "skipped": N}
    """
    counts = {"success": 0, "failed": 0, "skipped": 0}

    for move in plan.moves:
        if not move.src.exists():
            counts["skipped"] += 1
            continue

        ok = _safe_copy(move.src, move.dst)
        if ok:
            counts["success"] += 1
        else:
            counts["failed"] += 1

    return counts