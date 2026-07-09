# scanner.py
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
import screenshot_classifier
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────
IMAGE_EXTS = {
    ".apng", ".png", ".gif",
    ".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp",
    ".webp", ".tiff", ".tif",
    ".heif", ".heic", ".avif",
    ".eps", ".cr2", ".cr3", ".nef", ".arw"
}

VIDEO_EXTS = {
    ".webm", ".mkv", ".flv", ".vob", ".ogv", ".ogg",
    ".rrc", ".gifv", ".mng", ".mov", ".avi", ".qt",
    ".wmv", ".yuv", ".rm", ".asf", ".amv",
    ".mp4", ".m4p", ".m4v", ".mpg", ".mp2",
    ".mpeg", ".mpe", ".mpv", ".svi"
}

DOC_EXTS = {
    ".pdf", ".docx", ".txt", ".rtf", ".odt", ".pages",
    ".md", ".markdown",
    ".xlsx", ".csv", ".ods",
    ".pptx", ".key", ".odp",
    ".html", ".xml", ".json",
    ".epub", ".mobi", ".azw",
    ".svg"
}
# Common screen resolutions as (width, height) tuples
# Think: iPhone, Android, desktop monitors, tablets
SCREEN_RESOLUTIONS = {
    # Desktop
    (1920, 1080), (1080, 1920),
    (1366,  768), ( 768, 1366),
    (1536,  864), ( 864, 1536),
    (1280,  720), ( 720, 1280),
    (1600,  900), ( 900, 1600),
    (2560, 1440), (1440, 2560),
    (3840, 2160), (2160, 3840),
    # Tablet
    ( 768, 1024), (1024,  768),
    (1280,  800), ( 800, 1280),
    (1024,  768), ( 768, 1024),
    # Mobile
    ( 360,  640), ( 640,  360),
    ( 375,  812), ( 812,  375),
    ( 390,  844), ( 844,  390),
    ( 412,  915), ( 915,  412),
    ( 414,  896), ( 896,  414),
    (1080, 2400), (2400, 1080),
    (1440, 3200), (3200, 1440),
    (1284, 2778), (2778, 1284),
    ( 828, 1792), (1792,  828),
    # Common iPhone older models
    (1170, 2532), (2532, 1170),
    (1080, 1920), (1920, 1080),
}


# ── Data model ─────────────────────────────────────────────────
@dataclass
class FileRecord:
    path:          Path
    ext:           str
    size_bytes:    int
    file_type:     str          # "image" | "video" | "document" | "screenshot" | "other"
    capture_date:  Optional[str] = None   # "YYYY-MM-DD" or None
    exif:          dict = field(default_factory=dict)
    is_screenshot: bool = False


# ── Private helpers ─────────────────────────────────────────────

def _read_exif(path: Path) -> dict:
    """
    Open image with Pillow, read raw EXIF, map tag IDs → tag names.
    Return {} on ANY failure — EXIF is unreliable, never let it crash.
    """
    try:
        with Image.open(path) as img:
            raw = img._getexif()          # returns {tag_id: value} or None
            if not raw:
                return {}
            # TAGS maps numeric id → human name e.g. {306: "DateTime"}
            return {TAGS.get(tag_id, tag_id): value for tag_id, value in raw.items()}
    except Exception:
        return {}


def _get_capture_date(exif: dict, path: Path) -> Optional[str]:
    """
    Return capture date as "YYYY-MM-DD" string, or None.

    Try in this order:
      1. EXIF keys: "DateTimeOriginal", "DateTime", "DateTimeDigitized"
         format is "YYYY:MM:DD HH:MM:SS" → slice and replace colons
      2. Regex the filename for 8 consecutive digits e.g. 20240115
         validate: year 1990-2100, month 01-12, day 01-31
      3. os.path.getmtime(path) → datetime → strftime
    """
    # 1. Try EXIF date fields in priority order
    for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
        value = exif.get(key)
        if value and isinstance(value, str) and len(value) >= 10:
            # Format is "YYYY:MM:DD HH:MM:SS" — grab first 10 chars and swap colons
            date_part = value[:10].replace(":", "-")
            # Basic sanity check: should look like YYYY-MM-DD
            if re.match(r"^\d{4}-\d{2}-\d{2}$", date_part):
                return date_part

    # 2. Regex the filename for 8 consecutive digits (e.g. 20240115)
    stem = path.stem
    match = re.search(r"(\d{8})", stem)
    if match:
        digits = match.group(1)
        year  = int(digits[0:4])
        month = int(digits[4:6])
        day   = int(digits[6:8])
        if 1990 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    # 3. Fall back to file modification time
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except Exception:
        return None


def _is_screenshot(path: Path, exif: dict, size: Optional[tuple] = None) -> bool:
    """
    Return True if the file is likely a screenshot.

    Check in this order (any one being True = screenshot):
      1. exif "Software" field contains "screenshot" or "snip" (case-insensitive)
      2. No camera "Make" in exif AND image size matches SCREEN_RESOLUTIONS
      3. Extension is .png AND no camera "Make" AND aspect ratio < 2.5
         AND size matches SCREEN_RESOLUTIONS
         aspect ratio = max(w,h) / min(w,h)
    """
    software = exif.get("Software", "")
    if isinstance(software, str):
        lower = software.lower()
        if "screenshot" in lower or "snip" in lower:
            return True

    has_camera_make = bool(exif.get("Make"))

    try:
        if size is None:
            with Image.open(path) as img:
                size = img.size
        w, h = size

        # 2. No camera make + resolution matches a known screen resolution
        if not has_camera_make and (w, h) in SCREEN_RESOLUTIONS:
            return True

        # 3. PNG + no camera make + aspect ratio < 2.5 + resolution match
        if path.suffix.lower() == ".png" and not has_camera_make:
            if min(w, h) > 0 and (w, h) in SCREEN_RESOLUTIONS:
                aspect = max(w, h) / min(w, h)
                if aspect < 2.5:
                    return True
    except Exception:
        pass

    return False

def _is_downloaded(path: Path, exif: dict) -> bool:
    # No camera make = not taken by a camera
    if not exif.get("Make"):
        return True
    # Heavily compressed — under 300KB suspicious for a real photo
    if path.stat().st_size < 300_000:
        return True
    return False

# ── Main entry point ────────────────────────────────────────────

def scan_folder(root: str) -> List[FileRecord]:
    """
    Recursively walk root, return a FileRecord for every non-hidden file.
    """
    records: List[FileRecord] = []

    # First pass — collect all file paths (fast, no processing)
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            all_files.append(Path(dirpath) / filename)

    # Second pass — process each file with a progress bar
    for full_path in tqdm(all_files, desc="Scanning files", unit="file"):
        # Skip files we can't stat
        try:
            stat = full_path.stat()
        except OSError:
            continue

        size_bytes = stat.st_size
        ext = full_path.suffix.lower()

        # Classify and build the FileRecord
        if ext in IMAGE_EXTS:
            exif = _read_exif(full_path)
            try:
                with Image.open(full_path) as img:
                    size = img.size
            except Exception:
                size = None

            # rule-based check
            rule_says_ss = _is_screenshot(full_path, exif, size)

            # ML check — falls back to None if model not trained yet
            ml_says_ss = screenshot_classifier.predict(full_path) == "screenshot"

            # Download check
            download_says_ss = _is_downloaded(full_path, exif)

            # either signal is enough to call it a screenshot
            is_ss = rule_says_ss or ml_says_ss or download_says_ss

            capture_date = _get_capture_date(exif, full_path)
            file_type = "screenshot" if is_ss else "image"
            records.append(FileRecord(
                path=full_path,
                ext=ext,
                size_bytes=size_bytes,
                file_type=file_type,
                capture_date=capture_date,
                exif=exif,
                is_screenshot=is_ss,
            ))

        elif ext in VIDEO_EXTS:
            capture_date = _get_capture_date({}, full_path)
            records.append(FileRecord(
                path=full_path,
                ext=ext,
                size_bytes=size_bytes,
                file_type="video",
                capture_date=capture_date,
            ))

        elif ext in DOC_EXTS:
            records.append(FileRecord(
                path=full_path,
                ext=ext,
                size_bytes=size_bytes,
                file_type="document",
            ))

        else:
            records.append(FileRecord(
                path=full_path,
                ext=ext,
                size_bytes=size_bytes,
                file_type="other",
            ))

    return records