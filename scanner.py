# scanner.py
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
from PIL.ExifTags import TAGS

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

    Hint: img._getexif() returns {tag_id: value}
          TAGS dict from PIL.ExifTags maps tag_id → human name
    """
    try:
        img = Image.open(path)        # open with Pillow
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
    pass


def _is_screenshot(path: Path, exif: dict) -> bool:
    """
    Return True if the file is likely a screenshot.

    Check in this order (any one being True = screenshot):
      1. exif "Software" field contains "screenshot" or "snip" (case-insensitive)
      2. No camera "Make" in exif AND image size matches SCREEN_RESOLUTIONS
      3. Extension is .png AND no camera "Make" AND aspect ratio < 2.5
         aspect ratio = max(w,h) / min(w,h)

    Wrap Image.open() in try/except — always.
    """
    pass


# ── Main entry point ────────────────────────────────────────────

def scan_folder(root: str) -> List[FileRecord]:
    """
    Recursively walk root, return a FileRecord for every non-hidden file.

    Steps:
      1. os.walk — for each dir, filter out hidden dirs in-place:
             dirnames[:] = [d for d in dirnames if not d.startswith('.')]
      2. Skip hidden files (startswith '.') and skip files you can't stat
      3. For each file:
           - get ext (lower), size from stat
           - IMAGE_EXTS  → _read_exif, _is_screenshot, _get_capture_date
                           file_type = "screenshot" if is_ss else "image"
           - VIDEO_EXTS  → _get_capture_date (no exif), file_type = "video"
           - DOC_EXTS    → file_type = "document"
           - else        → file_type = "other"
      4. Append FileRecord, return the full list
    """
    pass