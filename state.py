# state.py
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

STATE_FILE = "organised.json"


class State:
    """
    Persists across runs. Tracks:
      - processed files by MD5
      - face embeddings per named person
    """
    def __init__(self, output_root: str):
        self.state_path = Path(output_root) / STATE_FILE
        self.data = {
            "processed": {},   # {md5: {"original": path, "organised_to": path, "date": date}}
            "people": {}       # {"Mum": {"embeddings": [[...128d...]], "folder": "People/Mum"}}
        }
        self._load()

    def _load(self):
        """Load existing state from disk if it exists."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    loaded = json.load(f)
                self.data["processed"] = loaded.get("processed", {})
                self.data["people"] = loaded.get("people", {})
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        """Save current state to disk."""
        def _default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.data, f, indent=2, default=_default)

    # ── Processed files ───────────────────────────────────────

    def is_seen(self, md5: str) -> bool:
        """Return True if this MD5 has been processed before."""
        return md5 in self.data["processed"]

    def mark_seen(self, md5: str, original_path: str, organised_to: str):
        """Record a file as processed."""
        self.data["processed"][md5] = {
            "original": original_path,
            "organised_to": organised_to,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

    # ── People / face embeddings ──────────────────────────────

    def match_person(self, embedding: np.ndarray, threshold: float = 0.5) -> Optional[str]:
        """
        Compare embedding against all stored person embeddings.
        Return person name if match found, None if no match.
        """
        best_name = None
        best_distance = None

        for name, info in self.data["people"].items():
            for stored in info.get("embeddings", []):
                distance = np.linalg.norm(embedding - np.array(stored))
                if distance < threshold:
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_name = name

        return best_name

    def add_person(self, name: str, embedding: np.ndarray, folder: str):
        """
        Add a new person or add embedding to existing person.
        """
        if name not in self.data["people"]:
            self.data["people"][name] = {"embeddings": [], "folder": folder}
        self.data["people"][name]["embeddings"].append(embedding.tolist())

    def get_people(self) -> Dict:
        """Return all stored people."""
        return self.data.get("people", {})

    def get_person_folder(self, name: str) -> Optional[str]:
        """Return the folder path for a named person."""
        return self.data["people"].get(name, {}).get("folder")

    def rename_person(self, old_name: str, new_name: str, new_folder: str):
        """
        Update person name and folder in state.
        Called when user renames a People/ folder manually.
        """
        if old_name not in self.data["people"]:
            return

        info = self.data["people"].pop(old_name)
        info["folder"] = new_folder
        self.data["people"][new_name] = info