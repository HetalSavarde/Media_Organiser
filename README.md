# Media Organiser

A local-first CLI tool that automatically classifies, deduplicates, and sorts 
photos, videos, screenshots, and documents into a clean folder structure.
No cloud. No APIs. Originals are never deleted.

---

## What It Does

| Feature | How |
|---|---|
| Sorts by type | Photos, Videos, Screenshots, Documents, Other |
| Sorts by date | Year → Month from EXIF data |
| Exact duplicates | MD5 hash comparison |
| Near duplicates | Perceptual hashing (pHash) + BK-tree |
| Best shot picker | Sharpness (Laplacian variance) + exposure scoring |
| Screenshot detection | Trained Random Forest classifier — 93.4% accuracy |
| Person grouping | face_recognition 128-d embeddings + DBSCAN clustering |
| Safe copy | MD5-verified copy — originals never touched |

---

## Architecture

Single-pass pipeline. Each module has one job and speaks a shared 
data contract (`FileRecord`). Nothing touches the filesystem until 
the organiser executes a pre-built plan — making `--dry-run` trivially safe.

scanner.py → deduper.py → quality_scorer.py → face_clusterer.py → organiser.py
↑
screenshot_classifier.py  (trained Random Forest, ships pre-trained)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Image I/O | Pillow |
| Computer Vision | OpenCV |
| Face Detection | face_recognition (dlib) |
| Duplicate Detection | imagehash (pHash) + pybktree (BK-tree) |
| ML Classifier | scikit-learn RandomForestClassifier |
| Model Serialisation | joblib |
| Progress | tqdm + colorama |

---

## Key Engineering Decisions

**Plan before execute** — organiser builds a complete list of file 
operations before touching anything. `--dry-run` skips the execute 
call entirely. No partial states, no surprises.

**ML + rules combined** — screenshot classifier and rule-based EXIF 
detector both run; either signal is enough. If no model exists yet, 
rules handle it automatically — nothing breaks.

**BK-tree for near-duplicate search** — replaced O(n²) pairwise pHash 
comparison with a BK-tree, reducing lookup complexity to O(log n) 
using the triangle inequality on Hamming distance.

**128-d face embeddings** — uses dlib's deep learning model via 
face_recognition instead of Haar cascade pixel patches. Embeddings 
capture actual facial geometry — prevents wrong clustering across persons.

**Safe copy with MD5 verification** — every file is verified after 
copy before the source is considered done. Originals untouched until 
you manually delete them.

---

## Output Structure

Organised/
├── Photos/
│   └── 2024/
│       ├── January/
│       └── March/
├── Videos/
│   └── 2024/
├── Screenshots/
│   └── 2024/
├── Documents/
│   ├── PDF/
│   ├── Word/
│   ├── Spreadsheets/
│   ├── Presentations/
│   └── Text/
├── People/
│   ├── Person_1/
│   └── Person_2/
├── Duplicates/
│   ├── Exact/
│   └── Similar/
└── Other/

---

## Setup

### 1. Install Python 3.9+

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on dlib (face recognition):**
> dlib requires C++ build tools to compile.
> - Windows: install Visual Studio Build Tools first
> - Mac: `brew install cmake`
> - Linux: `sudo apt install cmake build-essential`

### 3. That's it
A pre-trained screenshot classifier ships in `models/`.
Run immediately — no training required.

---

## Usage

```bash
# Always dry-run first — preview without copying anything
python main.py --input ~/Pictures --output ~/Organised --dry-run

# Run for real
python main.py --input ~/Pictures --output ~/Organised

# Skip face clustering (faster)
python main.py --input ~/Pictures --output ~/Organised --skip-faces

# Skip duplicate detection
python main.py --input ~/Pictures --output ~/Organised --no-dedup
```

---

## Optional — Retrain the Screenshot Classifier

The shipped model was trained on one person's photo library.
For better accuracy on your specific screenshots:

```bash
# Add your images:
# data/screenshots/  ← 100+ screenshots
# data/real_photos/  ← 100+ real photos

python screenshot_classifier.py
```

Training completes in under a minute and saves a new
`models/screenshot_classifier.pkl`.

---

## After Running

1. Check `Duplicates/Exact/` — identical files, safe to delete
2. Check `Duplicates/Similar/` — near-dupes, best shot stayed in Photos/
3. Rename `People/Person_N/` folders to actual names after reviewing
4. Delete originals only once you're happy with the output

---

## Known Limitations

- Face detection works best on clear front-facing portraits
- Small or partially visible faces may be missed (HOG-based detection limit)
- Screenshot classifier trained on one person's data — retrain for best results
- Currently single-threaded — upgrade path is `concurrent.futures` for parallel scanning
