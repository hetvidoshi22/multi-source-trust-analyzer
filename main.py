# -*- coding: utf-8 -*-
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
main.py
─────────────────────────────────────────────
Multi-Source Data Scraping Pipeline
─────────────────────────────────────────────
Entry point that orchestrates:
  1. Blog scraper       → 3 blog posts
  2. YouTube scraper    → 2 YouTube videos
  3. PubMed scraper     → 1 PubMed article
  4. Combined output    → output/scraped_data.json
  5. Individual files   → output/scraped_data/blogs.json
                          output/scraped_data/youtube.json
                          output/scraped_data/pubmed.json

Usage:
  python main.py                  # Run all scrapers
  python main.py --blogs          # Run only blog scraper
  python main.py --youtube        # Run only YouTube scraper
  python main.py --pubmed         # Run only PubMed scraper
  python main.py --trust-demo     # Run trust score demo
"""

import json
import os
import sys
import argparse
import time
from datetime import datetime

# Ensure project root is on the path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from project.scraper.blog_scraper import scrape_all_blogs
from project.scraper.youtube_scraper import scrape_all_youtube
from project.scraper.pubmed_scraper import scrape_all_pubmed
from project.scoring.trust_score import get_score_breakdown


OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "scraped_data")
COMBINED_OUTPUT = os.path.join(ROOT_DIR, "output", "scraped_data.json")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def save_json(data: list, filepath: str) -> None:
    """Save data as formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved -> {filepath}  ({len(data)} records)")


def print_header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_summary(label: str, items: list) -> None:
    print(f"\n  [{label}] -- {len(items)} records scraped:")
    for item in items:
        score  = item.get("trust_score", "N/A")
        author = item.get("author", "Unknown")
        if len(author) > 45: author = author[:42] + "..."
        title  = item.get("title", item.get("source_url", ""))[:50]
        date   = item.get("published_date", "Unknown")
        chunks = len(item.get("content_chunks", []))
        tags   = ", ".join(item.get("topic_tags", [])[:3])
        lang   = item.get("language", "?")
        region = item.get("region", "?")
        breakdown = item.get("trust_score_breakdown", {})
        if breakdown:
            parts = [f"{k[:10]}={v:.2f}" for k, v in breakdown.items()]
            bd_str = "  [" + " | ".join(parts) + "]"
        else:
            bd_str = ""
        print(f"    * {title}")
        print(f"      Author: {author} | Date: {date} | Lang: {lang} | Region: {region}")
        print(f"      Trust: {score}{bd_str}")
        print(f"      Tags: [{tags}] | Chunks: {chunks}")


def run_trust_demo() -> None:
    """Print trust score breakdown for example sources."""
    print_header("Trust Score Demo — Component Breakdown")

    examples = [
        {
            "label": "PubMed Peer-Reviewed Article",
            "author": "John A. Smith; Jane B. Doe",
            "published_date": "2024-03-15",
            "domain": "pubmed.ncbi.nlm.nih.gov",
            "citation_count": 45,
            "has_medical_disclaimer": True,
            "source_type": "pubmed",
        },
        {
            "label": "Reputable Tech Blog (TDS)",
            "author": "Alex Turner",
            "published_date": "2025-01-10",
            "domain": "towardsdatascience.com",
            "citation_count": 0,
            "has_medical_disclaimer": False,
            "source_type": "blog",
        },
        {
            "label": "Generic Blog (Unknown Author)",
            "author": "Unknown",
            "published_date": "Unknown",
            "domain": "myblog123.blogspot.com",
            "citation_count": 0,
            "has_medical_disclaimer": False,
            "source_type": "blog",
        },
        {
            "label": "Spam Health Blog (Misinformation)",
            "author": "ADMIN123",
            "published_date": "2018-06-01",
            "domain": "natural-cures.xyz",
            "citation_count": 0,
            "has_medical_disclaimer": False,
            "source_type": "blog",
            "content_sample": "miracle cure doctors don't want you to know",
        },
        {
            "label": "YouTube Educational Channel",
            "author": "3Blue1Brown",
            "published_date": "2023-09-20",
            "domain": "youtube.com",
            "citation_count": 0,
            "has_medical_disclaimer": False,
            "source_type": "youtube",
        },
    ]

    for ex in examples:
        label = ex.pop("label")
        breakdown = get_score_breakdown(**ex)
        print(f"\n  +- {label}")
        print(f"  |  Raw Scores:")
        for comp, val in breakdown["raw_scores"].items():
            weight = breakdown["weights"].get(comp, 0)
            weighted = breakdown["weighted_scores"].get(comp, 0)
            bar = "#" * int(val * 20) + "-" * (20 - int(val * 20))
            print(f"  |    {comp:<30} {bar}  {val:.3f} x {weight:.2f} = {weighted:.4f}")
        print(f"  +- FINAL TRUST SCORE: {breakdown['total_trust_score']:.3f}")


# ---------------------------------------------
# MAIN RUNNER
# ---------------------------------------------

def run_pipeline(run_blogs=True, run_youtube=True, run_pubmed=True) -> None:
    start_time = time.time()
    print_header("Multi-Source Data Scraping Pipeline")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_results = []

    # ── Blogs ──────────────────────────────────
    if run_blogs:
        print_header("Step 1/3: Scraping Blog Posts")
        blogs = scrape_all_blogs()
        print_summary("Blogs", blogs)
        save_json(blogs, os.path.join(OUTPUT_DIR, "blogs.json"))
        all_results.extend(blogs)

    # ── YouTube ────────────────────────────────
    if run_youtube:
        print_header("Step 2/3: Scraping YouTube Videos")
        videos = scrape_all_youtube()
        print_summary("YouTube", videos)
        save_json(videos, os.path.join(OUTPUT_DIR, "youtube.json"))
        all_results.extend(videos)

    # ── PubMed ─────────────────────────────────
    if run_pubmed:
        print_header("Step 3/3: Scraping PubMed Articles")
        articles = scrape_all_pubmed()
        print_summary("PubMed", articles)
        save_json(articles, os.path.join(OUTPUT_DIR, "pubmed.json"))
        all_results.extend(articles)

    # ── Combined output ────────────────────────
    if all_results:
        save_json(all_results, COMBINED_OUTPUT)

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE | Records: {} | Time: {:.1f}s".format(len(all_results), elapsed))
    print("  Output Directory: {}".format(os.path.join(ROOT_DIR, 'output')))
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Source Data Scraping Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blogs",      action="store_true", help="Run only blog scraper")
    parser.add_argument("--youtube",    action="store_true", help="Run only YouTube scraper")
    parser.add_argument("--pubmed",     action="store_true", help="Run only PubMed scraper")
    parser.add_argument("--trust-demo", action="store_true", help="Show trust score demo")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.trust_demo:
        run_trust_demo()
        sys.exit(0)

    # If specific flags provided, run only those
    specific = args.blogs or args.youtube or args.pubmed
    run_pipeline(
        run_blogs=args.blogs   if specific else True,
        run_youtube=args.youtube if specific else True,
        run_pubmed=args.pubmed if specific else True,
    )
