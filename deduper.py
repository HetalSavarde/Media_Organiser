# deduper.py
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import imagehash
from PIL import Image
from scanner import FileRecord
import pybktree

PHASH_THRESHOLD = 8  # hamming distance — 0 = identical, ≤8 = very similar


# ── Helpers ────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    """
    Read file in chunks and return its MD5 hash as a hex string.
    Return "" on failure.
    """
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


def _phash(path: Path):
    """
    Return the perceptual hash of an image using imagehash.phash().
    Return None on failure.
    """
    try:
        img = Image.open(path).convert("RGB")
        return imagehash.phash(img)
    except Exception:
        return None


# ── Main function ──────────────────────────────────────────────

def find_duplicates(records: List[FileRecord]) -> Tuple[
    Dict[str, List[FileRecord]],   # exact_groups  {md5: [records]}
    List[List[FileRecord]]          # near_groups   [[records], ...]
]:
    """
    Two-stage duplicate detection on image FileRecords only.
    """
    # ── Stage 1: Exact duplicates via MD5 ─────────────────────

    # Filter to images and screenshots only
    image_records = [r for r in records if r.file_type in ("image", "screenshot")]

    md5_groups: Dict[str, List[FileRecord]] = defaultdict(list)
    for record in image_records:
        digest = _md5(record.path)
        if digest:  # skip files that failed to hash
            md5_groups[digest].append(record)

    # Only keep groups with more than one file
    exact_groups = {digest: group for digest, group in md5_groups.items() if len(group) > 1}

    # Track paths already accounted for as exact duplicates
    exact_paths = {r.path for group in exact_groups.values() for r in group}

    # ── Stage 2: Near duplicates via pHash ────────────────────

    # Only process images not already in an exact group
    remaining = [r for r in image_records if r.path not in exact_paths]

    # Compute pHash for each remaining image
    phashes: List[Tuple[FileRecord, object]] = []
    for record in remaining:
        ph = _phash(record.path)
        if ph is not None:
            phashes.append((record, ph))

    # build the tree — uses hamming distance between pHashes
    tree = pybktree.BKTree(lambda a, b: a[1] - b[1], phashes)

    visited = set()
    near_groups: List[List[FileRecord]] = []

    for i, (record, ph) in enumerate(phashes):
        if i in visited:
            continue
        # find ALL hashes within threshold distance in one lookup
        matches = tree.find((record, ph), PHASH_THRESHOLD)
        if len(matches) <= 1:
            continue
        cluster = []
        for _dist, (matched_record, _ph) in matches:
            idx = next(k for k, (r, _) in enumerate(phashes) if r.path == matched_record.path)
            if idx not in visited:
                cluster.append(matched_record)
                visited.add(idx)
        if len(cluster) > 1:
            near_groups.append(cluster)
    return exact_groups, near_groups