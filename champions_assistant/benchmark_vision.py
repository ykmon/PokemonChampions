"""
Benchmark tool to compare original vs enhanced vision engine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from champions_assistant.config import AppConfig, load_config
from champions_assistant.data_loader import DataRepository
from champions_assistant.templates import PokemonTemplateMatcher


@dataclass
class BenchmarkResult:
    method: Literal["original", "enhanced"]
    species_id: str | None
    confidence: float
    duration_ms: float
    accepted: bool
    template_path: Path | None = None


def benchmark_image(image_path: Path, repository: DataRepository) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Compare original vs enhanced matching on a single image"""
    image_bytes = image_path.read_bytes()

    # Original method
    start = time.perf_counter()
    matcher_original = PokemonTemplateMatcher(
        repository,
        use_enhanced_matching=False,
    )
    match_original = matcher_original.match(image_bytes)
    duration_original = (time.perf_counter() - start) * 1000

    result_original = BenchmarkResult(
        method="original",
        species_id=match_original.species_id,
        confidence=match_original.confidence,
        duration_ms=duration_original,
        accepted=match_original.accepted,
        template_path=match_original.template_path,
    )

    # Enhanced method
    start = time.perf_counter()
    matcher_enhanced = PokemonTemplateMatcher(
        repository,
        use_enhanced_matching=True,
        enable_preprocessing=True,
        enable_verification=True,
    )
    match_enhanced = matcher_enhanced.match(image_bytes)
    duration_enhanced = (time.perf_counter() - start) * 1000

    result_enhanced = BenchmarkResult(
        method="enhanced",
        species_id=match_enhanced.species_id,
        confidence=match_enhanced.confidence,
        duration_ms=duration_enhanced,
        accepted=match_enhanced.accepted,
        template_path=match_enhanced.template_path,
    )

    return result_original, result_enhanced


def benchmark_directory(
    directory: Path,
    repository: DataRepository,
    *,
    expected_species: dict[str, str] | None = None,
) -> None:
    """
    Benchmark all images in a directory.

    Args:
        directory: Path to directory containing test images
        repository: Data repository
        expected_species: Optional mapping of filename -> expected species_id for accuracy check
    """
    image_files = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))

    if not image_files:
        print(f"No images found in {directory}")
        return

    print(f"\n{'='*80}")
    print(f"Benchmarking {len(image_files)} images from: {directory}")
    print(f"{'='*80}\n")

    original_results: list[BenchmarkResult] = []
    enhanced_results: list[BenchmarkResult] = []

    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {image_path.name}...", end=" ", flush=True)

        try:
            result_original, result_enhanced = benchmark_image(image_path, repository)
            original_results.append(result_original)
            enhanced_results.append(result_enhanced)

            # Print quick comparison
            status = "✓" if result_original.species_id == result_enhanced.species_id else "⚠"
            print(f"{status} Original: {result_original.confidence:.3f} | Enhanced: {result_enhanced.confidence:.3f}")

        except Exception as e:
            print(f"✗ Error: {e}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    # Speed comparison
    avg_original = sum(r.duration_ms for r in original_results) / len(original_results)
    avg_enhanced = sum(r.duration_ms for r in enhanced_results) / len(enhanced_results)
    speedup = avg_original / avg_enhanced if avg_enhanced > 0 else 0

    print("⏱️  PERFORMANCE:")
    print(f"  Original:  {avg_original:.1f} ms/image")
    print(f"  Enhanced:  {avg_enhanced:.1f} ms/image")
    print(f"  Speedup:   {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}\n")

    # Confidence comparison
    avg_conf_original = sum(r.confidence for r in original_results) / len(original_results)
    avg_conf_enhanced = sum(r.confidence for r in enhanced_results) / len(enhanced_results)

    print("🎯 CONFIDENCE:")
    print(f"  Original:  {avg_conf_original:.3f} average")
    print(f"  Enhanced:  {avg_conf_enhanced:.3f} average")
    print(f"  Improvement: {avg_conf_enhanced - avg_conf_original:+.3f}\n")

    # Acceptance rate
    accept_original = sum(1 for r in original_results if r.accepted) / len(original_results) * 100
    accept_enhanced = sum(1 for r in enhanced_results if r.accepted) / len(enhanced_results) * 100

    print("✅ ACCEPTANCE RATE:")
    print(f"  Original:  {accept_original:.1f}%")
    print(f"  Enhanced:  {accept_enhanced:.1f}%")
    print(f"  Improvement: {accept_enhanced - accept_original:+.1f}%\n")

    # Accuracy check if ground truth provided
    if expected_species:
        correct_original = 0
        correct_enhanced = 0

        for img_file in image_files:
            expected = expected_species.get(img_file.stem)
            if not expected:
                continue

            orig_match = next((r for r in original_results if r.template_path and img_file.name in str(r.template_path)), None)
            enh_match = next((r for r in enhanced_results if r.template_path and img_file.name in str(r.template_path)), None)

            if orig_match and orig_match.species_id == expected:
                correct_original += 1
            if enh_match and enh_match.species_id == expected:
                correct_enhanced += 1

        total = len(expected_species)
        if total > 0:
            print("📊 ACCURACY (vs ground truth):")
            print(f"  Original:  {correct_original}/{total} = {correct_original/total*100:.1f}%")
            print(f"  Enhanced:  {correct_enhanced}/{total} = {correct_enhanced/total*100:.1f}%\n")

    # Detailed disagreements
    disagreements = [
        (orig, enh) for orig, enh in zip(original_results, enhanced_results)
        if orig.species_id != enh.species_id
    ]

    if disagreements:
        print(f"⚠️  DISAGREEMENTS: {len(disagreements)} cases where methods differ\n")
        for idx, (orig, enh) in enumerate(disagreements[:5], 1):  # Show first 5
            print(f"  {idx}. Original: {orig.species_id} ({orig.confidence:.3f}) | "
                  f"Enhanced: {enh.species_id} ({enh.confidence:.3f})")
        if len(disagreements) > 5:
            print(f"  ... and {len(disagreements) - 5} more")
    else:
        print("✓ No disagreements - both methods agree on all images")

    print(f"\n{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark vision engine improvements")
    parser.add_argument("--image", type=Path, help="Single image to benchmark")
    parser.add_argument("--directory", type=Path, help="Directory of images to benchmark")
    parser.add_argument("--dataset", type=Path, help="Dataset directory with manifest.json for ground truth")
    args = parser.parse_args()

    config = load_config()
    repository = DataRepository(config.data_dir)

    if args.image:
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            return 1

        result_original, result_enhanced = benchmark_image(args.image, repository)

        print(f"\n{'='*60}")
        print(f"Image: {args.image.name}")
        print(f"{'='*60}\n")
        print(f"Original Method:")
        print(f"  Species:    {result_original.species_id}")
        print(f"  Confidence: {result_original.confidence:.4f}")
        print(f"  Accepted:   {result_original.accepted}")
        print(f"  Duration:   {result_original.duration_ms:.1f} ms\n")
        print(f"Enhanced Method:")
        print(f"  Species:    {result_enhanced.species_id}")
        print(f"  Confidence: {result_enhanced.confidence:.4f}")
        print(f"  Accepted:   {result_enhanced.accepted}")
        print(f"  Duration:   {result_enhanced.duration_ms:.1f} ms\n")

        if result_original.species_id == result_enhanced.species_id:
            print("✓ Both methods agree")
        else:
            print("⚠ Methods disagree!")

    elif args.directory:
        if not args.directory.is_dir():
            print(f"Error: Directory not found: {args.directory}")
            return 1
        benchmark_directory(args.directory, repository)

    elif args.dataset:
        # Load ground truth from dataset manifest
        import json
        manifest_path = args.dataset / "manifest.json"
        if not manifest_path.exists():
            print(f"Error: manifest.json not found in {args.dataset}")
            return 1

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {}
        for entry in manifest.get("images", []):
            filename = Path(entry["path"]).stem
            expected[filename] = entry.get("species_id")

        images_dir = args.dataset / "images"
        if not images_dir.is_dir():
            images_dir = args.dataset

        benchmark_directory(images_dir, repository, expected_species=expected)

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python -m champions_assistant.benchmark_vision --image screenshots/preview.png")
        print("  python -m champions_assistant.benchmark_vision --directory screenshots/")
        print("  python -m champions_assistant.benchmark_vision --dataset dataset/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
