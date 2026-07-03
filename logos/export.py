"""Export all logo SVGs to multiple PNG resolutions."""

import subprocess
import sys
from pathlib import Path

LOGOS_DIR = Path(__file__).parent

CONCEPTS = ["concept-a", "concept-b", "concept-c", "concept-d"]

ICON_SIZES = [16, 32, 48, 64, 128, 256, 512]
WORDMARK_WIDTHS = [480, 960, 1440]


def export_svg(svg_path: Path, output_path: Path, size: int, *, is_wordmark: bool = False):
    """Export SVG to PNG using rsvg-convert."""
    if is_wordmark:
        cmd = ["rsvg-convert", "-w", str(size), str(svg_path), "-o", str(output_path)]
    else:
        cmd = ["rsvg-convert", "-w", str(size), "-h", str(size), str(svg_path), "-o", str(output_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL: {output_path.name} — {result.stderr.strip()}")
        return False
    print(f"  {output_path.name}")
    return True


def main():
    for concept in CONCEPTS:
        concept_dir = LOGOS_DIR / concept
        logo_svg = concept_dir / "logo.svg"
        wordmark_svg = concept_dir / "wordmark.svg"

        print(f"\n{concept}:")

        if logo_svg.exists():
            for size in ICON_SIZES:
                if size <= 48:
                    out_dir = concept_dir / "favicon"
                else:
                    out_dir = concept_dir / "icon"
                out_dir.mkdir(parents=True, exist_ok=True)
                export_svg(logo_svg, out_dir / f"logo-{size}x{size}.png", size)

        if wordmark_svg.exists():
            out_dir = concept_dir / "wordmark"
            out_dir.mkdir(parents=True, exist_ok=True)
            for width in WORDMARK_WIDTHS:
                export_svg(wordmark_svg, out_dir / f"wordmark-{width}w.png", width, is_wordmark=True)


if __name__ == "__main__":
    main()
