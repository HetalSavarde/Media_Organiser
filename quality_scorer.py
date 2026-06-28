# quality_scorer.py
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from scanner import FileRecord


# ── Scoring helpers ────────────────────────────────────────────

def _sharpness(path: Path) -> float:
    """
    Measure how in-focus the image is using Laplacian variance.
    Higher = sharper. Return 0.0 on failure.
    """
    try:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _exposure(path: Path) -> float:
    """
    Score how well-exposed the image is. Return 0.0 to 1.0.
    Penalise very dark (mean < 50) or blown out (mean > 220).
    Ideal range is 80-180 → score 1.0.
    """
    try:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.0
        mean = float(gray.mean())
        if mean < 50 or mean > 220:
            return 0.3
        if 80 <= mean <= 180:
            return 1.0
        return 0.7
    except Exception:
        return 0.0


def score(record: FileRecord) -> float:
    """
    Combined quality score for one image.
    Sharpness is weighted heavily, exposure is a tie-breaker.
    Return: sharpness * exposure
    """
    return _sharpness(record.path) * _exposure(record.path)


# ── Main function ──────────────────────────────────────────────

def pick_best(group: List[FileRecord]) -> Tuple[FileRecord, List[FileRecord]]:
    """
    Given a list of similar images, return (best, rest).
    best = highest scoring image → stays in Photos/
    rest = everything else → goes to Duplicates/Similar/
    """
    scored = sorted(group, key=score, reverse=True)
    return scored[0], scored[1:]