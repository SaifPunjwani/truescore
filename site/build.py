# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Render the method docs to HTML for the published site.

The pages are written as markdown so they read well in the repository, which is where most
people meet them. Deploying them unchanged serves them as plain text, so the four pages
carrying the actual derivations are unreadable in a browser and invisible to search. This
converts them, reusing the landing page's stylesheet so there is one source of styling,
and rewrites inter-page links from .md to .html.

Run from the repository root:

    python site/build.py _site
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import markdown

BASE = "https://saifpunjwani.github.io/truescore"

STUDIES = {
    "mt_bench": (
        "mt-bench",
        "88% agreement, 13 points of error",
        "GPT-4 agrees with human judges on 88% of MT-Bench comparisons and still reports "
        "its own win rate 12.7 points above theirs. Measured on public data, reproducible.",
    ),
}

PAGES = {
    "README": ("Methods", "Derivations for the estimators in truescore."),
    "prediction-powered-inference": (
        "Prediction-powered inference",
        "Correcting an LLM-judge metric with a small gold subset: the estimator, the "
        "variance-optimal lambda, and the Rogan-Gladen alternative.",
    ),
    "confidence-sequences": (
        "Confidence sequences",
        "Why a fixed-sample confidence interval fails under repeated inspection, and two "
        "time-uniform constructions that do not.",
    ),
    "judge-bias-and-slices": (
        "Judge bias and per-segment correction",
        "HC3-robust regression of judge error, position bias, and why one global "
        "correction cannot fix a judge whose bias varies by segment.",
    ),
    "contamination": (
        "Benchmark contamination testing",
        "An exact permutation test for a memorised evaluation set, with a false-positive "
        "rate equal to the level by construction.",
    ),
}

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - truescore</title>
<meta name="description" content="{description}">
<link rel="canonical" href="{canonical}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:type" content="article">
<meta property="og:url" content="{canonical}">
<meta property="og:image" content="https://opengraph.githubassets.com/1/SaifPunjwani/truescore">
<meta name="twitter:card" content="summary_large_image">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
      integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+"
      crossorigin="anonymous">
<style>
{style}
main {{ max-width: 46rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
main table {{ border-collapse: collapse; margin: 1.25rem 0; font-size: 0.95rem; }}
main th, main td {{ border: 1px solid var(--line); padding: 0.4rem 0.7rem; text-align: left; }}
main blockquote {{ border-left: 3px solid var(--line); margin: 1rem 0; padding-left: 1rem; color: var(--muted); }}
.crumb {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 1.5rem; }}
.crumb a {{ color: var(--accent); }}
</style>
</head>
<body>
<main>
<p class="crumb"><a href="{base}/">truescore</a> / <a href="{base}/docs/methods/">methods</a></p>
{body}
</main>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
        integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg"
        crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        integrity="sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk"
        crossorigin="anonymous"
        onload="renderMathInElement(document.body, {{delimiters: [
          {{left: '$$', right: '$$', display: true}},
          {{left: '$', right: '$', display: false}}]}});"></script>
</body>
</html>
"""


def stylesheet(root: Path) -> str:
    """Reuse the landing page's CSS so the docs cannot drift away from it visually."""
    text = (root / "site" / "index.html").read_text(encoding="utf-8")
    match = re.search(r"<style>(.*?)</style>", text, re.S)
    if match is None:
        raise SystemExit("site/index.html has no <style> block to share")
    return match.group(1).strip()


def convert(text: str) -> str:
    """Markdown to HTML, with inter-page .md links rewritten for the web."""
    text = re.sub(r"\]\((?!https?:)([\w./-]+)\.md\)", r"](\1.html)", text)
    return markdown.markdown(
        text, extensions=["tables", "fenced_code", "attr_list", "sane_lists"]
    )


def render(root: Path, out: Path) -> int:
    style = stylesheet(root)
    written = 0

    target = out / "docs" / "methods"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted((root / "docs" / "methods").glob("*.md")):
        stem = path.stem
        title, description = PAGES.get(stem, (stem.replace("-", " ").capitalize(), ""))
        name = "index" if stem == "README" else stem
        canonical = f"{BASE}/docs/methods/" + ("" if name == "index" else f"{name}.html")
        (target / f"{name}.html").write_text(
            TEMPLATE.format(
                title=html.escape(title),
                description=html.escape(description),
                canonical=canonical,
                base=BASE,
                style=style,
                body=convert(path.read_text(encoding="utf-8")),
            ),
            encoding="utf-8",
        )
        written += 1

    findings = out / "findings"
    findings.mkdir(parents=True, exist_ok=True)
    for directory, (slug, title, description) in STUDIES.items():
        path = root / "analysis" / directory / "FINDINGS.md"
        if not path.exists():
            raise SystemExit(f"study {directory} has no FINDINGS.md")
        (findings / f"{slug}.html").write_text(
            TEMPLATE.format(
                title=html.escape(title),
                description=html.escape(description),
                canonical=f"{BASE}/findings/{slug}.html",
                base=BASE,
                style=style,
                body=convert(path.read_text(encoding="utf-8")),
            ),
            encoding="utf-8",
        )
        written += 1
    return written


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    root = Path(__file__).resolve().parent.parent
    out = Path(argv[1])
    count = render(root, out)
    print(f"rendered {count} method pages into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
