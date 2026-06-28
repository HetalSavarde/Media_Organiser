# screenshot_classifier.py
import os
import numpy as np
import cv2
import joblib
from pathlib import Path
from typing import Optional
from PIL import Image

MODEL_PATH = Path("models/screenshot_classifier.pkl")
DATA_DIR   = Path("data")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_clf_cache: object = None  # loaded once, reused across all predict() calls


# ── Feature extraction ─────────────────────────────────────────

def extract_features(image_path: Path) -> Optional[np.ndarray]:
    """
    Convert one image into a fixed-length feature vector.
    Returns None if the image can't be read.
    """
    try:
        gray  = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if gray is None or color is None:
            return None

        h, w = gray.shape
        total_pixels = h * w

        # 1. aspect_ratio — max / min dimension
        aspect_ratio = max(w, h) / min(w, h)

        # 2. pixel_variance — flat screenshots have low variance, photos are noisy
        pixel_variance = float(np.var(gray))

        # 3. edge_density — ratio of Canny edge pixels to total pixels
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / total_pixels

        # 4. unique_colours — normalised unique RGB tuples
        #    Cap at 50,000 pixels for speed by sampling if needed
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        pixels = color_rgb.reshape(-1, 3)
        if len(pixels) > 50_000:
            idx = np.random.choice(len(pixels), 50_000, replace=False)
            pixels = pixels[idx]
        sample_size = len(pixels)
        unique_colours = len(set(map(tuple, pixels))) / sample_size

        # 5. brightness_mean — average grayscale value
        brightness_mean = float(np.mean(gray))

        # 6. brightness_std — std dev; screenshots are bimodal (dark bg + bright text)
        brightness_std = float(np.std(gray))

        return np.array([
            aspect_ratio,
            pixel_variance,
            edge_density,
            unique_colours,
            brightness_mean,
            brightness_std,
        ], dtype=np.float32)

    except Exception:
        return None


# ── Training ───────────────────────────────────────────────────

def train(data_dir: Path = DATA_DIR, model_path: Path = MODEL_PATH):
    """
    Train the screenshot classifier and save the model.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    X, y = [], []

    # label 1 = screenshot
    ss_dir = data_dir / "screenshots"
    print(f"Loading screenshots from {ss_dir} ...")
    for root, _, files in os.walk(ss_dir):
        for fname in files:
            if Path(fname).suffix.lower() not in VALID_EXTS:
                continue
            path = Path(root) / fname
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(1)
    print(f"  Loaded {sum(1 for label in y if label == 1)} screenshots")

    # label 0 = real photo
    photo_dir = data_dir / "real_photos"
    print(f"Loading real photos from {photo_dir} ...")
    before = len(y)
    for root, _, files in os.walk(photo_dir):
        for fname in files:
            if Path(fname).suffix.lower() not in VALID_EXTS:
                continue
            path = Path(root) / fname
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(0)
    print(f"  Loaded {len(y) - before} real photos")
    print(f"Total samples: {len(y)}")

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} ...")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["real_photo", "screenshot"]))

    feature_names = [
        "aspect_ratio", "pixel_variance", "edge_density",
        "unique_colours", "brightness_mean", "brightness_std",
    ]
    print("Feature importances:")
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:<20} {imp:.3f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")


# ── Prediction ─────────────────────────────────────────────────

def predict(image_path: Path, model_path: Path = MODEL_PATH) -> Optional[str]:
    """
    Predict whether image_path is a "screenshot" or "real_photo".
    Returns None if model not found or features can't be extracted.
    """
    global _clf_cache

    if not model_path.exists():
        return None

    if _clf_cache is None:
        _clf_cache = joblib.load(model_path)

    features = extract_features(image_path)
    if features is None:
        return None

    result = _clf_cache.predict([features])[0]
    return "screenshot" if result == 1 else "real_photo"


# ── Entry point for training ────────────────────────────────────

if __name__ == "__main__":
    train()