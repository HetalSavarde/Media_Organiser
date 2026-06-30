# main.py
import sys
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from colorama import init as colorama_init, Fore, Style
from scanner import scan_folder
from deduper import find_duplicates
from quality_scorer import pick_best
from face_clusterer import cluster_faces
from organiser import build_plan, execute

colorama_init(autoreset=True)


# ── Print helpers ──────────────────────────────────────────────

def hdr(text):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def ok(text):
    print(f"  {Fore.GREEN}✓{Style.RESET_ALL}  {text}")

def info(text):
    print(f"  {Fore.YELLOW}→{Style.RESET_ALL}  {text}")


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local Photo Organiser")
    parser.add_argument("--input", required=True, help="Source folder to scan")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-faces", action="store_true", help="Skip face clustering")
    parser.add_argument("--no-dedup", action="store_true", help="Skip duplicate detection")
    args = parser.parse_args()

    try:
        # ── Validate input folder ───────────────────────────
        input_path = Path(args.input)
        if not input_path.exists() or not input_path.is_dir():
            print(f"{Fore.RED}Error: input folder '{args.input}' does not exist{Style.RESET_ALL}")
            sys.exit(1)

        # ── Step 1: Scan ─────────────────────────────────────
        hdr("Step 1: Scanning files")
        records = scan_folder(args.input)
        ok(f"Found {len(records)} files")

        type_counts = {}
        for rec in records:
            type_counts[rec.file_type] = type_counts.get(rec.file_type, 0) + 1
        for file_type, count in sorted(type_counts.items()):
            info(f"{file_type}: {count}")

        # ── Step 2: Duplicate detection ──────────────────────
        exact_dupes = {}
        near_dupes  = []
        if args.no_dedup:
            hdr("Step 2: Duplicate detection")
            info("Skipped (--no-dedup)")
        else:
            hdr("Step 2: Duplicate detection")
            exact_dupes, near_dupes_raw = find_duplicates(records)
            exact_count = sum(len(g) - 1 for g in exact_dupes.values())
            ok(f"Found {len(exact_dupes)} exact duplicate groups ({exact_count} extra files)")
            ok(f"Found {len(near_dupes_raw)} near duplicate groups")

            # ── Step 3: Pick best in each near-dupe group ────
            near_dupes = []
            for group in near_dupes_raw:
                best, rest = pick_best(group)
                near_dupes.append([best] + rest)

        # ── Step 4: Face clustering ──────────────────────────
        face_clusters = {}
        if args.skip_faces:
            hdr("Step 4: Face clustering")
            info("Skipped (--skip-faces)")
        else:
            hdr("Step 4: Face clustering")
            face_clusters = cluster_faces(records)
            real_clusters = {k: v for k, v in face_clusters.items() if k != -1}
            ok(f"Found {len(real_clusters)} people clusters")
            noise_count = len(face_clusters.get(-1, []))
            if noise_count:
                info(f"{noise_count} faces unclustered (noise)")

        # ── Step 5: Build plan ────────────────────────────────
        hdr("Step 5: Building organisation plan")
        plan = build_plan(records, exact_dupes, near_dupes, face_clusters, args.output)
        ok(f"{len(plan.moves)} planned operations")
        print(plan.summary())

        # ── Step 6: Dry-run or execute ────────────────────────
        if args.dry_run:
            hdr("Dry run — no files were copied")
            info(f"Would copy {len(plan.moves)} files to '{args.output}'")
        else:
            hdr("Step 6: Executing plan")
            from organiser import _safe_copy

            results = {"success": 0, "failed": 0, "skipped": 0}
            for move in tqdm(plan.moves, desc="Copying files", unit="file"):
                if not move.src.exists():
                    results["skipped"] += 1
                    continue
                if _safe_copy(move.src, move.dst):
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    
            ok(f"Success: {results['success']}")
            if results["failed"]:
                print(f"  {Fore.RED}✗{Style.RESET_ALL}  Failed: {results['failed']}")
            if results["skipped"]:
                info(f"Skipped: {results['skipped']}")

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted — exiting cleanly.{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()