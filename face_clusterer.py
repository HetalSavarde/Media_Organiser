# face_clusterer.py
import cv2
import numpy as np
import face_recognition
from pathlib import Path
from typing import List, Dict
from sklearn.cluster import DBSCAN
from scanner import FileRecord

DBSCAN_EPS   = 0.5   # max distance between two face embeddings to be same person
DBSCAN_MIN   = 2     # minimum photos to form a person cluster


# ── Face detection ─────────────────────────────────────────────

def _detect_faces(path: Path) -> List[np.ndarray]:
    """
    Detect all faces in an image and return 128-d embedding vectors.
    One vector per face found. Return [] on failure or no faces.

    face_recognition works on RGB images (not BGR like OpenCV).
    """
    try:
        img = face_recognition.load_image_file(str(path))
        # returns list of 128-d numpy arrays — one per face found
        encodings = face_recognition.face_encodings(img)
        return encodings
    except Exception:
        return []


# ── Main function ──────────────────────────────────────────────

def cluster_faces(records: List[FileRecord]) -> Dict[int, List[FileRecord]]:
    """
    Detect faces in all image FileRecords and cluster them by person.
    Returns {cluster_id: [FileRecord, ...]}
    cluster_id -1 = faces that didn't fit any cluster (DBSCAN noise)
    """
    # 1. Images only — no screenshots, videos, or documents
    image_records = [r for r in records if r.file_type == "image"]

    # 2. Detect faces — take first face per image
    face_data: List[tuple] = []   # (FileRecord, 128-d embedding)
    for record in image_records:
        encodings = _detect_faces(record.path)
        if encodings:
            face_data.append((record, encodings[0]))

    # 3. Need at least 2 faces to cluster
    if len(face_data) < 2:
        return {}

    # 4. Build feature matrix — no normalisation needed
    #    face_recognition embeddings are already unit normalised
    face_records = [fd[0] for fd in face_data]
    X = np.array([fd[1] for fd in face_data])

    # 5. Run DBSCAN
    #    0.5 euclidean distance ≈ 0.6 face_recognition tolerance
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN, metric="euclidean")
    labels = db.fit_predict(X)

    # 6. Group records by cluster label
    clusters: Dict[int, List[FileRecord]] = {}
    for record, label in zip(face_records, labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(record)

    return clusters