# face_clusterer.py
import cv2
import numpy as np
import face_recognition
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.cluster import DBSCAN
from scanner import FileRecord
from tqdm import tqdm

DBSCAN_EPS   = 0.5
DBSCAN_MIN   = 2


# ── Face detection ─────────────────────────────────────────────

def _detect_faces(path: Path) -> List[np.ndarray]:
    try:
        img = face_recognition.load_image_file(str(path))
        h, w = img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return face_recognition.face_encodings(img)
    except Exception:
        return []


# ── Main function ──────────────────────────────────────────────

def cluster_faces(records: List[FileRecord], state=None) -> Dict[str, List[FileRecord]]:
    """
    Detect faces and cluster by person.
    
    If state is provided:
      - new faces are matched against previously seen people first
      - matched faces go to existing person folders
      - unmatched faces go through DBSCAN as new people
    
    Returns {person_name: [FileRecord, ...]}
    Key is now a string name not an int — either "Mum", "Person_1" etc.
    -1 key still used for noise.
    """
    # 1. Images only
    image_records = [r for r in records if r.file_type == "image"]

    # 2. Detect faces
    face_data: List[tuple] = []   # (FileRecord, 128-d embedding)
    for record in tqdm(image_records, desc="Detecting faces", unit="img"):
        encodings = _detect_faces(record.path)
        if encodings:
            face_data.append((record, encodings[0]))

    if not face_data:
        return {}

    # 3. If state exists — match against previously known people first
    unmatched_face_data = []
    clusters: Dict = {}

    if state is not None:
        for record, embedding in face_data:
            match = state.match_person(embedding)
            if match:
                # known person — route to their existing folder
                if match not in clusters:
                    clusters[match] = []
                clusters[match].append(record)
            else:
                # unknown — needs clustering
                unmatched_face_data.append((record, embedding))
    else:
        unmatched_face_data = face_data

    # 4. Cluster unmatched faces with DBSCAN
    if len(unmatched_face_data) >= 2:
        face_records = [fd[0] for fd in unmatched_face_data]
        X = np.array([fd[1] for fd in unmatched_face_data])

        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN, metric="euclidean")
        labels = db.fit_predict(X)

        # find next available Person number
        existing_numbers = []
        if state:
            for name in state.get_people():
                if name.startswith("Person_"):
                    try:
                        existing_numbers.append(int(name.split("_")[1]))
                    except ValueError:
                        pass
        next_num = max(existing_numbers, default=0) + 1

        # map DBSCAN label → person name
        label_to_name: Dict[int, str] = {}
        for label in set(labels):
            if label == -1:
                label_to_name[-1] = -1   # noise stays as -1
            else:
                label_to_name[label] = f"Person_{next_num}"
                next_num += 1

        for record, label in zip(face_records, labels):
            name = label_to_name[int(label)]
            if name not in clusters:
                clusters[name] = []
            clusters[name].append(record)

        # 5. Save new people to state
        if state is not None:
            for (record, embedding), label in zip(unmatched_face_data, labels):
                if label == -1:
                    continue
                name = label_to_name[int(label)]
                folder = f"People/{name}"
                state.add_person(name, embedding, folder)

    elif len(unmatched_face_data) == 1:
        # only one unmatched face — can't cluster, goes to noise
        clusters.setdefault(-1, []).append(unmatched_face_data[0][0])

    return clusters